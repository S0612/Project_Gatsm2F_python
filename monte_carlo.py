r"""Monte Carlo study of finite sample bias in the two factor GATSM.

For each replication a yield sample is simulated from the model at the
"true" parameters, the model is then re-estimated by maximum likelihood, and the
estimates are collected. Bias is the mean of the estimates minus the truth.

As in the thesis, every replication is estimated from two starting points
(the true parameters and a PCA based start) and the better likelihood is
kept. Starting at the truth is a computational shortcut: it means the study
measures bias around the true optimum, not the optimiser's ability to find it.

Everything is seeded, so results are reproducible and do not depend on the
number of worker processes.

The default number of simulations is 1000, set by "--R". By default the true
parameters are the estimates saved by estimate.py in results/params_estimated.json
("--params"). "--truth thesis" uses the thesis values in gatsm.START_VALUES instead.

The code uses all processing cores by default. Change this with "--workers".

"--summarise results/mc_estimates.npy" rebuilds the table and figure from saved
estimates without simulating again.

To run (Spyder), use the Run button (F5), or paste this into the IPython console:
    %run monte_carlo.py --params results/params_estimated.json --R 1000
where --params is the file saved by estimate.py.

In other IDEs, run the command in CMD or PowerShell set to the gatsm2f folder:
    .venv\Scripts\python monte_carlo.py --params results\params_estimated.json --R 1000
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
 
import numpy as np
import pandas as pd
 
import gatsm as g
import plotting as pl
 
_STATE = {}
 
 
def _init(true_dict, mats, T, starts):
    _STATE.update(p=g.params_from_dict(true_dict), mats=mats, T=T, starts=starts)
 
 
def _one(job):
    r, seed = job
    t0 = time.time()
    p, mats, T, starts = _STATE["p"], _STATE["mats"], _STATE["T"], _STATE["starts"]
    y, _ = g.simulate(p, mats, T, np.random.default_rng(seed))
    fits = []
    if "true" in starts:
        fits.append(g.fit(y, mats, p))
    if "pca" in starts:
        try:
            fits.append(g.fit(y, mats, g.pca_start(y)))
        except Exception:
            pass
    ok = [f for f in fits if f.params is not None and f.nll < g.PENALTY]
    if not ok:
        return r, np.full(len(g.PARAM_NAMES), np.nan), np.nan, False, time.time() - t0
    best = min(ok, key=lambda f: f.nll)
    return r, g.params_to_vector(best.params), best.nll, True, time.time() - t0
 
 
def summarise(est, true_vec, T, out):
    """Bias table and figure from an (R, P) array of estimates.
 
    Failed replications are rows of NaN and are left out. Bias (%) is left blank for
    parameters whose true value is (close to) zero, where a relative bias means nothing.
    """
    good = est[~np.isnan(est).any(axis=1)]
    n = len(good)
    mean, med, sd = good.mean(axis=0), np.median(good, axis=0), good.std(axis=0, ddof=1)
    bias = mean - true_vec
    rel = np.where(np.abs(true_vec) > 1e-4, 100 * bias / np.where(np.abs(true_vec) > 1e-4, true_vec, 1.0), np.nan)
    table = pd.DataFrame({"True": true_vec, "Mean": mean, "Median": med, "Std": sd, "Bias": bias,
                          "Bias (%)": rel, "Bias / MC s.e.": bias / (sd / np.sqrt(n))},
                         index=g.PARAM_NAMES)
    print(f"\nMonte Carlo finite sample bias, {n} of {len(est)} replications used")
    print(table.to_string(float_format=lambda v: f"{v:12.6f}"))
    table.to_csv(out / "mc_bias_table.csv")
 
    pl.mc_histograms(
        good, true_vec,
        ["phiP11", "phiP22", "phiQ11", "phiQ22", "alpha", "stdY"],
        ["Persistence of factor 1 (phiP11)", "Persistence of factor 2 (phiP22)",
         "Risk neutral eigenvalue 1 (phiQ1)", "Risk neutral eigenvalue 2 (phiQ2)",
         "Short rate intercept (alpha)", "Measurement error (stdY)"],
        out / "mc_distributions.png",
        title=f"Monte Carlo estimates, R = {n} replications, T = {T} months",
    )
    print(f"Saved bias table and figure in {out}/")
 
 
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default="results/params_estimated.json",
                    help="JSON written by estimate.py. If missing, the thesis starting values are used.")
    ap.add_argument("--truth", choices=["estimates", "thesis"], default="estimates",
                    help="true parameters: the estimates in --params (default) or the thesis values "
                         "in gatsm.START_VALUES")
    ap.add_argument("--R", type=int, default=1000, help="number of replications")
    ap.add_argument("--T", type=int, default=648, help="months per simulated sample")
    ap.add_argument("--mats", type=float, nargs="+", default=list(g.DEFAULT_MATURITIES_YEARS))
    ap.add_argument("--starts", nargs="+", default=["true", "pca"], choices=["true", "pca"])
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2)))
    ap.add_argument("--seed", type=int, default=456)
    ap.add_argument("--out", default="results")
    ap.add_argument("--summarise", default=None, metavar="NPY",
                    help="skip the simulation and rebuild the table and figure from a saved mc_estimates.npy")
    args = ap.parse_args()
 
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
 
    pfile = Path(args.params)
    if args.truth == "thesis":
        true_dict = dict(g.START_VALUES)
        print("True parameters: the thesis values in gatsm.START_VALUES (diagonal P dynamics).")
    elif pfile.exists():
        saved = json.loads(pfile.read_text())
        true_dict = saved["params"]
        print(f"True parameters: estimates in {pfile}, estimated on {saved.get('yields_file', 'an unknown data file')} "
              f"(average log likelihood {saved.get('avg_loglik', float('nan')):.6f})")
    else:
        true_dict = dict(g.START_VALUES)
        print("*" * 78)
        print(f"{pfile} not found: using the thesis STARTING values as the true parameters.")
        print("To use your own estimates instead, run estimate.py first or fix the --params path.")
        print("*" * 78)
    mats = np.array([int(round(m * 12)) for m in args.mats])
    p_true = g.params_from_dict(true_dict)
    if not g.stationary(p_true):
        raise SystemExit("The true parameters are not stationary under P.")
    true_vec = g.params_to_vector(p_true)
 
    if args.summarise:
        summarise(np.load(args.summarise), true_vec, args.T, out)
        return
 
    seeds = np.random.SeedSequence(args.seed).spawn(args.R)
    jobs = list(enumerate(seeds))
    print(f"R = {args.R}, T = {args.T}, maturities (years) {args.mats}, starts {args.starts}, "
          f"{args.workers} worker(s)")
 
    est = np.full((args.R, len(g.PARAM_NAMES)), np.nan)
    nll = np.full(args.R, np.nan)
    conv = np.zeros(args.R, dtype=bool)
    secs = np.zeros(args.R)
    t0 = time.time()
    done = 0
    with mp.Pool(args.workers, initializer=_init, initargs=(true_dict, mats, args.T, args.starts)) as pool:
        for r, vec, f, ok, dt in pool.imap_unordered(_one, jobs, chunksize=1):
            est[r], nll[r], conv[r], secs[r] = vec, f, ok, dt
            done += 1
            if done % max(1, args.R // 10) == 0 or done == args.R:
                print(f"  {done}/{args.R} replications, {time.time() - t0:.0f} s elapsed", flush=True)
    print(f"Converged: {conv.sum()} / {args.R}.  Mean time per replication {secs.mean():.1f} s, "
          f"wall clock {(time.time() - t0) / 60:.1f} min")
 
    np.save(out / "mc_estimates.npy", est)
    summarise(est, true_vec, args.T, out)
 
 
if __name__ == "__main__":
    mp.freeze_support()
    main()