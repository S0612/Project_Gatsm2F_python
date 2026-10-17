"""Checks of gatsm.py against independent, deliberately simple implementations."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gatsm as g  # noqa: E402

MATS = np.array(g.DEFAULT_MATURITIES_YEARS) * 12


def start_params():
    return g.params_from_dict(g.START_VALUES)


def test_loadings_match_recursion():
    """Closed form loadings equal the loop recursion used in the MATLAB code."""
    p = start_params()
    nmax = MATS.max()
    beta = np.ones(2)
    PhiQ = np.diag(p.phiQ)
    S2 = p.Sigma @ p.Sigma.T
    A = np.zeros(nmax)
    B = np.zeros((2, nmax))
    for k in range(1, nmax + 1):
        if k == 1:
            A[0] = -p.alpha
            B[:, 0] = -beta
        else:
            A[k - 1] = -p.alpha + A[k - 2] + 0.5 * B[:, k - 2] @ S2 @ B[:, k - 2]
            B[:, k - 1] = -beta + PhiQ.T @ B[:, k - 2]
    g0_ref = 12 * (-A[MATS - 1] / MATS)
    gx_ref = 12 * (-B[:, MATS - 1].T / MATS[:, None])
    g0, gx, _, _ = g.loadings(p, MATS)
    assert np.allclose(g0, g0_ref, rtol=1e-12, atol=1e-14)
    assert np.allclose(gx, gx_ref, rtol=1e-12, atol=1e-14)


def naive_kalman_loglik(p, y):
    """Textbook Kalman filter with no steady state shortcut."""
    g0, gx, _, _ = g.loadings(p, MATS)
    N = y.shape[1]
    Q = p.Sigma @ p.Sigma.T
    x = np.linalg.solve(np.eye(2) - p.Phi, p.mu)
    P = np.linalg.solve(np.eye(4) - np.kron(p.Phi, p.Phi), Q.reshape(-1)).reshape(2, 2)
    R = p.stdY ** 2 * np.eye(N)
    total = 0.0
    for t in range(len(y)):
        xp = p.mu + p.Phi @ x
        Pp = p.Phi @ P @ p.Phi.T + Q
        F = gx @ Pp @ gx.T + R
        v = y[t] - g0 - gx @ xp
        Finv = np.linalg.inv(F)
        total += -0.5 * (N * np.log(2 * np.pi) + np.linalg.slogdet(F)[1] + v @ Finv @ v)
        Kg = Pp @ gx.T @ Finv
        x = xp + Kg @ v
        P = Pp - Kg @ F @ Kg.T
    return total


def test_filter_matches_naive_kalman():
    p = start_params()
    y, _ = g.simulate(p, MATS, 648, np.random.default_rng(1))
    fast = g.loglik(p, y, MATS)
    slow = naive_kalman_loglik(p, y)
    assert abs(fast - slow) < 1e-6 * abs(slow), (fast, slow)


def test_simulation_moments():
    """Simulated states have the model's unconditional mean and covariance."""
    p = start_params()
    # use a less persistent model so that a long sample is informative
    p.Phi = np.array([[0.9, 0.0], [0.0, 0.8]])
    p.mu = np.array([1e-4, -1e-4])
    Y, X = g.simulate(p, MATS, 200000, np.random.default_rng(2))
    x0, P0, _ = g._unconditional(p)
    assert np.allclose(X.mean(axis=0), x0, atol=5e-4)
    assert np.allclose(np.cov(X.T), P0, rtol=0.05)


def test_pack_unpack_roundtrip():
    p = start_params()
    q = g.unpack(g.pack(p))
    assert np.allclose(g.params_to_vector(p), g.params_to_vector(q), rtol=1e-9, atol=1e-12)


def test_likelihood_peaks_at_truth_on_average():
    """Perturbing the true parameters lowers the likelihood on a long sample."""
    p = start_params()
    y, _ = g.simulate(p, MATS, 5000, np.random.default_rng(3))
    base = g.loglik(p, y, MATS)
    for name, factor in [("stdY", 1.5), ("sigma11", 1.3), ("alpha", 1.2)]:
        d = g.params_to_dict(p)
        d[name] *= factor
        assert g.loglik(g.params_from_dict(d), y, MATS) < base


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
