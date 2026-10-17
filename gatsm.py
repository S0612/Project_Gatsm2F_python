"""Two factor latent Gaussian affine term structure model (GATSM).

Python version of the estimation and Monte Carlo code from my master's thesis
"Survey Expectations and Finite-Sample Bias in a Macro-Finance Gaussian Affine
Term Structure Model" (MATLAB original: 2F_codes/).

Model (monthly time step, yields quoted as annualised decimals):

    short rate     r_t     = 12 * (alpha + beta' x_t),   beta = (1, 1)
    P dynamics     x_{t+1} = mu + Phi x_t + Sigma eps_{t+1},   eps ~ N(0, I)
    Q dynamics     x_{t+1} = diag(phiQ) x_t + Sigma eps^Q_{t+1}   (muQ = 0)
    observations   y_t     = g0 + gx x_t + stdY * v_t,           v ~ N(0, I)

The two risk neutral eigenvalues are ordered phiQ1 >= phiQ2, which is the
normalisation used in the thesis. Parameters are estimated by maximum
likelihood, with the likelihood evaluated by a Kalman filter.
"""


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

try:
    from numba import njit
except ImportError:  # pure Python fallback: same results, much slower
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(f):
            return f

        return decorator

K = 2  # number of latent factors
DEFAULT_MATURITIES_YEARS = (1, 2, 5, 7, 10, 15)

# Same order as the parameter vector in GATSM_US_2F_V2.m
PARAM_NAMES = [
    "phiQ11", "phiQ22", "alpha", "muP1", "muP2",
    "phiP11", "phiP12", "phiP21", "phiP22",
    "sigma11", "sigma21", "sigma22", "stdY",
]

# Starting values from StartingParams2F_US.m (in the MATLAB estimation script).
# They are starting values used by our given optimiser, the simple Nelder-Mead in this case. It is not the final estimates.
START_VALUES = {
    "phiQ11": 0.9970249269, "phiQ22": 0.9719181887, "alpha": 0.0040417407,
    "muP1": 0.0000055757, "muP2": -0.0000626359,
    "phiP11": 0.9970249269, "phiP12": 0.0, "phiP21": 0.0, "phiP22": 0.9719181887,
    "sigma11": 0.0003164546, "sigma21": 0.0, "sigma22": 0.0001822510,
    "stdY": 0.0010356332,
}


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class Params:
    phiQ: np.ndarray    # (K,) diagonal of the risk neutral AR matrix, descending
    alpha: float        # monthly short rate intercept
    mu: np.ndarray      # (K,) P mean
    Phi: np.ndarray     # (K, K) P AR matrix
    Sigma: np.ndarray   # (K, K) lower triangular Cholesky factor
    stdY: float         # measurement error std (annualised decimal units)


def params_from_dict(d: dict) -> Params:
    return Params(
        phiQ=np.array([d["phiQ11"], d["phiQ22"]], dtype=float),
        alpha=float(d["alpha"]),
        mu=np.array([d["muP1"], d["muP2"]], dtype=float),
        Phi=np.array([[d["phiP11"], d["phiP12"]], [d["phiP21"], d["phiP22"]]], dtype=float),
        Sigma=np.array([[d["sigma11"], 0.0], [d["sigma21"], d["sigma22"]]], dtype=float),
        stdY=float(d["stdY"]),
    )


def params_to_dict(p: Params) -> dict:
    return {
        "phiQ11": float(p.phiQ[0]), "phiQ22": float(p.phiQ[1]), "alpha": float(p.alpha),
        "muP1": float(p.mu[0]), "muP2": float(p.mu[1]),
        "phiP11": float(p.Phi[0, 0]), "phiP12": float(p.Phi[0, 1]),
        "phiP21": float(p.Phi[1, 0]), "phiP22": float(p.Phi[1, 1]),
        "sigma11": float(p.Sigma[0, 0]), "sigma21": float(p.Sigma[1, 0]),
        "sigma22": float(p.Sigma[1, 1]), "stdY": float(p.stdY),
    }


def params_to_vector(p: Params) -> np.ndarray:
    d = params_to_dict(p)
    return np.array([d[n] for n in PARAM_NAMES])


def params_from_vector(v: np.ndarray) -> Params:
    return params_from_dict(dict(zip(PARAM_NAMES, v)))


def pack(p: Params) -> np.ndarray:
    """Natural parameters -> unconstrained vector used by the optimiser."""
    q = np.clip(p.phiQ, 1e-6, 1 - 1e-9)
    return np.array([
        np.log(q[0] / (1 - q[0])), np.log(q[1] / (1 - q[1])),
        p.alpha, p.mu[0], p.mu[1],
        p.Phi[0, 0], p.Phi[0, 1], p.Phi[1, 0], p.Phi[1, 1],
        np.log(p.Sigma[0, 0]), p.Sigma[1, 0], np.log(p.Sigma[1, 1]),
        np.log(p.stdY),
    ])


def unpack(theta: np.ndarray) -> Params | None:
    """Unconstrained vector -> Params. Returns None if the vector is not usable."""
    if not np.all(np.isfinite(theta)) or np.any(np.abs(theta[[0, 1, 9, 11, 12]]) > 40):
        return None
    phiQ = 1.0 / (1.0 + np.exp(-theta[0:2]))
    phiQ = np.sort(phiQ)[::-1]  # normalisation: phiQ1 >= phiQ2
    return Params(
        phiQ=phiQ,
        alpha=float(theta[2]),
        mu=np.array(theta[3:5], dtype=float),
        Phi=np.array(theta[5:9], dtype=float).reshape(2, 2),
        Sigma=np.array([[np.exp(theta[9]), 0.0], [theta[10], np.exp(theta[11])]]),
        stdY=float(np.exp(theta[12])),
    )


# ---------------------------------------------------------------------------
# Bond pricing
# ---------------------------------------------------------------------------

def loadings(p: Params, mats_months: np.ndarray):
    """Yield loadings g0, gx (annualised) and short rate loadings r0, rx.

    Closed form of the pricing recursion
        B_k = -beta + PhiQ' B_{k-1},
        A_k = -alpha + A_{k-1} + 0.5 B_{k-1}' Sigma Sigma' B_{k-1}
    for a diagonal PhiQ and muQ = 0. The yield of maturity n months is
    y = 12 * (-A_n / n) + 12 * (-B_n / n)' x.
    """
    mats_months = np.asarray(mats_months, dtype=int)
    nmax = int(mats_months.max())
    k = np.arange(nmax)[:, None]
    S = np.cumsum(p.phiQ[None, :] ** k, axis=0)       # S_k = sum_{i<k} phiQ^i, row k-1
    B = -S                                             # beta = (1, 1)
    S2 = p.Sigma @ p.Sigma.T
    q = 0.5 * np.einsum("ki,ij,kj->k", B, S2, B)       # 0.5 B_k' S2 B_k
    A = -p.alpha * np.arange(1, nmax + 1) + np.concatenate(([0.0], np.cumsum(q)[:-1]))
    idx = mats_months - 1
    n = mats_months.astype(float)
    g0 = 12.0 * (-A[idx] / n)
    gx = 12.0 * (-B[idx] / n[:, None])
    r0 = 12.0 * p.alpha
    rx = 12.0 * np.ones(K)
    return g0, gx, r0, rx


# ---------------------------------------------------------------------------
# Kalman filter
# ---------------------------------------------------------------------------

@njit(cache=True)
def _kalman(y, g0, gx, gxT, R, mu, Phi, PhiT, Q, x0, P0, tol):
    """Kalman filter for y_t = g0 + gx x_t + noise, x_t = mu + Phi x_{t-1} + shock.

    Once the state covariance has converged (relative change below tol) the
    gain and the innovation covariance are constant, so the filter switches
    to the steady state recursion. This changes the likelihood by a negligible
    amount and makes each evaluation about as fast as a few matrix products.
    """
    T, N = y.shape
    Kd = mu.shape[0]
    logl = np.zeros(T)
    xhat = np.zeros((T, Kd))
    x = x0.copy()
    P = P0.copy()
    xp = np.zeros(Kd)
    v = np.zeros(N)
    Finv = np.zeros((N, N))
    Kg = np.zeros((Kd, N))
    logdet = 0.0
    steady = False
    t_steady = -1
    c = N * np.log(2.0 * np.pi)
    for t in range(T):
        for i in range(Kd):
            s = mu[i]
            for j in range(Kd):
                s += Phi[i, j] * x[j]
            xp[i] = s
        for n in range(N):
            s = y[t, n] - g0[n]
            for i in range(Kd):
                s -= gx[n, i] * xp[i]
            v[n] = s
        if not steady:
            Pp = Phi @ P @ PhiT + Q
            F = gx @ Pp @ gxT + R
            sign, logdet = np.linalg.slogdet(F)
            if sign <= 0.0 or not np.isfinite(logdet):
                return False, logl, xhat, t_steady
            Finv = np.linalg.inv(F)
            Kg = Pp @ gxT @ Finv
            Pnew = Pp - Kg @ gx @ Pp
            Pnew = 0.5 * (Pnew + Pnew.T)
            diff = np.max(np.abs(Pnew - P))
            scale = np.max(np.abs(Pnew))
            P = Pnew
            if t > 0 and diff <= tol * scale:
                steady = True
                t_steady = t
        q = 0.0
        for a in range(N):
            s = 0.0
            for b in range(N):
                s += Finv[a, b] * v[b]
            q += v[a] * s
        logl[t] = -0.5 * (c + logdet + q)
        for i in range(Kd):
            s = xp[i]
            for n in range(N):
                s += Kg[i, n] * v[n]
            x[i] = s
            xhat[t, i] = s
    return True, logl, xhat, t_steady


def _unconditional(p: Params):
    """Unconditional mean and covariance of the state under P."""
    Q = p.Sigma @ p.Sigma.T
    x0 = np.linalg.solve(np.eye(K) - p.Phi, p.mu)
    vecP = np.linalg.solve(np.eye(K * K) - np.kron(p.Phi, p.Phi), Q.reshape(-1))
    P0 = vecP.reshape(K, K)
    return x0, 0.5 * (P0 + P0.T), Q


def stationary(p: Params) -> bool:
    return bool(np.max(np.abs(np.linalg.eigvals(p.Phi))) < 1.0)


def run_filter(p: Params, y: np.ndarray, mats_months: np.ndarray, tol: float = 1e-12):
    """Run the Kalman filter. Returns (ok, logl per period, filtered states)."""
    if not stationary(p):
        return False, None, None
    g0, gx, _, _ = loadings(p, mats_months)
    N = y.shape[1]
    x0, P0, Q = _unconditional(p)
    R = (p.stdY ** 2) * np.eye(N)
    ok, logl, xhat, _ = _kalman(
        np.ascontiguousarray(y, dtype=float), g0, gx, np.ascontiguousarray(gx.T),
        R, p.mu, p.Phi, np.ascontiguousarray(p.Phi.T), Q, x0, P0, tol,
    )
    if not ok or not np.all(np.isfinite(logl)):
        return False, None, None
    return True, logl, xhat


def loglik(p: Params, y: np.ndarray, mats_months: np.ndarray) -> float:
    """Total log likelihood, or -inf if the parameters are not admissible."""
    ok, logl, _ = run_filter(p, y, mats_months)
    return float(logl.sum()) if ok else -np.inf


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@njit(cache=True)
def _simulate_states(mu, Phi, shocks, x1):
    T, Kd = shocks.shape
    X = np.zeros((T, Kd))
    X[0, :] = x1
    for t in range(1, T):
        for i in range(Kd):
            s = mu[i]
            for j in range(Kd):
                s += Phi[i, j] * X[t - 1, j]
            X[t, i] = s + shocks[t, i]
    return X


def simulate(p: Params, mats_months: np.ndarray, T: int, rng: np.random.Generator):
    """Simulate T months of yields (T x N) and states (T x K) from the model."""
    g0, gx, _, _ = loadings(p, mats_months)
    x0, P0, _ = _unconditional(p)
    x1 = x0 + np.linalg.cholesky(P0) @ rng.standard_normal(K)
    shocks = rng.standard_normal((T, K)) @ p.Sigma.T
    X = _simulate_states(p.mu, p.Phi, shocks, x1)
    Y = g0 + X @ gx.T + p.stdY * rng.standard_normal((T, len(mats_months)))
    return Y, X


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

# Initial simplex step sizes in the unconstrained parameter space (see pack()).
_STEPS = np.array([0.2, 0.2, 5e-4, 2e-5, 2e-5, 0.02, 0.02, 0.02, 0.02, 0.2, 5e-5, 0.2, 0.2])

PENALTY = 1e10


def neg_loglik_theta(theta: np.ndarray, y: np.ndarray, mats_months: np.ndarray) -> float:
    p = unpack(theta)
    if p is None:
        return PENALTY
    ll = loglik(p, y, mats_months)
    return -ll if np.isfinite(ll) else PENALTY


def _simplex(theta: np.ndarray) -> np.ndarray:
    sim = np.vstack([theta] + [theta + _STEPS[i] * np.eye(len(theta))[i] for i in range(len(theta))])
    return sim


@dataclass
class FitResult:
    params: Params
    theta: np.ndarray
    nll: float          # negative total log likelihood
    nfev: int
    converged: bool


def fit(y: np.ndarray, mats_months: np.ndarray, start: Params, restarts: int = 4,
        maxiter: int = 20000, xatol: float = 1e-8, fatol: float = 1e-7) -> FitResult:
    """Maximum likelihood by Nelder-Mead, restarted until the likelihood stops improving."""
    theta = pack(start)
    best_theta = theta
    best_f = neg_loglik_theta(theta, y, mats_months)
    nfev = 0
    converged = False
    for _ in range(restarts + 1):
        res = minimize(
            neg_loglik_theta, best_theta, args=(y, mats_months), method="Nelder-Mead",
            options=dict(initial_simplex=_simplex(best_theta), maxiter=maxiter,
                         maxfev=10 * maxiter, xatol=xatol, fatol=fatol, adaptive=True),
        )
        nfev += res.nfev
        improved = best_f - res.fun
        if res.fun < best_f:
            best_theta, best_f = res.x, res.fun
        if improved < 1e-6:
            converged = best_f < PENALTY
            break
    p = unpack(best_theta)
    return FitResult(params=p, theta=best_theta, nll=float(best_f), nfev=nfev, converged=converged)


def pca_start(y: np.ndarray) -> Params:
    """Rough starting values from principal components, as in the thesis Monte Carlo.

    The first K principal components of the yields stand in for the latent
    factors; a VAR(1) fitted to them gives mu, Phi and Sigma. The scale of
    these proxy factors differs from the model's factors, so this start is
    deliberately rough and is used together with a start at the true values.
    """
    Yd = y - y.mean(axis=0)
    U, s, _ = np.linalg.svd(Yd, full_matrices=False)
    X = U[:, :K] * s[:K]
    Z = np.column_stack([np.ones(len(X) - 1), X[:-1]])
    coef, *_ = np.linalg.lstsq(Z, X[1:], rcond=None)
    mu0, Phi0 = coef[0], coef[1:].T
    resid = X[1:] - Z @ coef
    Sigma0 = np.linalg.cholesky(np.atleast_2d(np.cov(resid.T)))
    rho = np.max(np.abs(np.linalg.eigvals(Phi0)))
    if rho >= 0.999:  # keep the start inside the stationary region
        Phi0 = Phi0 * (0.99 / rho)
    eig = np.sort(np.real(np.linalg.eigvals(Phi0)))[::-1]
    phiQ0 = np.clip(eig, 0.5, 0.999)
    # Sigma0 must have a positive diagonal for the log transform
    Sigma0 = np.array([[abs(Sigma0[0, 0]), 0.0], [Sigma0[1, 0], abs(Sigma0[1, 1])]])
    return Params(phiQ=phiQ0, alpha=float(y[:, 0].mean() / 12.0), mu=mu0, Phi=Phi0,
                  Sigma=Sigma0, stdY=0.012)


# ---------------------------------------------------------------------------
# Standard errors and post estimation
# ---------------------------------------------------------------------------

def opg_standard_errors(p: Params, y: np.ndarray, mats_months: np.ndarray,
                        rel_step: float = 1e-5) -> np.ndarray:
    """Outer product of scores (OPG) standard errors in natural parameter units."""
    v = params_to_vector(p)
    T = y.shape[0]
    scores = np.zeros((T, len(v)))
    for i in range(len(v)):
        h = rel_step * max(abs(v[i]), 1e-4)
        up, dn = v.copy(), v.copy()
        up[i] += h
        dn[i] -= h
        _, l_up, _ = run_filter(params_from_vector(up), y, mats_months)
        _, l_dn, _ = run_filter(params_from_vector(dn), y, mats_months)
        _, l_0, _ = run_filter(p, y, mats_months)
        if l_up is not None and l_dn is not None:
            scores[:, i] = (l_up - l_dn) / (2 * h)
        elif l_up is not None:
            scores[:, i] = (l_up - l_0) / h
        elif l_dn is not None:
            scores[:, i] = (l_0 - l_dn) / h
        else:
            scores[:, i] = np.nan
    scores = scores - scores.mean(axis=0)
    try:
        return np.sqrt(np.diag(np.linalg.inv(scores.T @ scores)))
    except np.linalg.LinAlgError:
        return np.full(len(v), np.nan)


def decompose(p: Params, xhat: np.ndarray, mats_months: np.ndarray):
    """Fitted yields, expected short rates and term premia (all annualised decimals).

    The expected short rate for maturity n is the average of E_t[r_{t+i}],
    i = 0..n-1, under P, using the filtered states.
    """
    g0, gx, r0, rx = loadings(p, mats_months)
    yhat = g0 + xhat @ gx.T
    nmax = int(np.max(mats_months))
    X = xhat.copy()
    running = np.zeros(len(xhat))
    avg = {}
    for i in range(1, nmax + 1):
        if i > 1:
            X = p.mu + X @ p.Phi.T
        running += r0 + X @ rx
        avg[i] = running / i
    rexp = np.column_stack([avg[int(n)] for n in mats_months])
    return yhat, rexp, yhat - rexp
