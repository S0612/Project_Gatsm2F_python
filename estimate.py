"""Estimate the two factor GATSM by maximum likelihood on a yield data set.

E.g.:
    python estimate.py --yields US_monthly_yields_Jan1972_Dec2025.csv \
                       --maturities US_monthly_yields_Jan1972_Dec2025_maturities.csv
                       
I use Spyder so I specifically need to run:
    %run estimate.py --yields US_monthly_yields_Jan1972_Dec2025.csv --start 1972-01

And since I've named my maturity data to that of my yield data with the affix "_maturities" it is automatically picked up.
Otherwise do as the first example to input your maturities.

Writes parameter estimates with OPG standard errors, pricing errors and the
yield decomposition (fitted yield, expected short rate, term premium).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import gatsm as g
import plotting as pl


def load_data(yields_csv, mats_csv, mats_years):
    raw = np.loadtxt(yields_csv, delimiter=",")
    avail = np.rint(np.atleast_1d(np.loadtxt(mats_csv, delimiter=",")).ravel() * 12).astype(int)
    cols = []
    for m in mats_years:
        hit = np.where(avail == int(round(m * 12)))[0]
        if len(hit) == 0:
            raise SystemExit(f"Maturity {m} years not found in {mats_csv}")
        cols.append(hit[0])
    y = raw[:, cols] / 100.0                       # percent -> annualised decimal
    if not np.all(np.isfinite(y)):
        raise SystemExit("Missing values in the selected yields are not supported.")
    return y, np.array([int(round(m * 12)) for m in mats_years])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yields", required=True,
                    help="CSV with the yields, e.g. US_monthly_yields_Jan1972_Dec2025.csv ")
    ap.add_argument("--maturities", default=None,
                    help="default: <yields file name>_maturities.csv")
    ap.add_argument("--mats", type=float, nargs="+", default=list(g.DEFAULT_MATURITIES_YEARS),
                    help="maturities in years to use (default: 1 2 5 7 10 15)")
    ap.add_argument("--start", default="1972-01", help="date of the first observation (YYYY-MM)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    yields_path = Path(args.yields)
    mats_path = Path(args.maturities) if args.maturities else yields_path.with_name(yields_path.stem + "_maturities.csv")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    y, mats = load_data(yields_path, mats_path, args.mats)
    T = len(y)
    dates = pd.date_range(args.start, periods=T, freq="MS")
    print(f"Data file: {yields_path}")
    print(f"{T} monthly observations, maturities (years): {[float(m) / 12 for m in mats]}")

    # Two starting points: the thesis starting values and a PCA based start.
    # If both reach the same likelihood, the optimum is not an artefact of the start.
    fits = {"thesis start": g.fit(y, mats, g.params_from_dict(g.START_VALUES))}
    try:
        fits["PCA start"] = g.fit(y, mats, g.pca_start(y))
    except Exception as exc:  # pragma: no cover
        print("PCA start failed:", exc)
    for name, f in fits.items():
        print(f"  {name:13s} avg log likelihood {-f.nll / T:.6f}  ({f.nfev} evaluations)")
    best_name = min(fits, key=lambda k: fits[k].nll)
    best = fits[best_name]
    p = best.params
    print(f"Using the fit from the {best_name}.")

    se = g.opg_standard_errors(p, y, mats)
    est = g.params_to_vector(p)

    table = pd.DataFrame({"Estimate": est, "Std. error": se, "t-stat": est / se}, index=g.PARAM_NAMES)
    print("\nParameter estimates (OPG standard errors)")
    print(table.to_string(float_format=lambda v: f"{v:12.6f}"))
    table.to_csv(out / "estimates.csv")

    ok, logl, xhat = g.run_filter(p, y, mats)
    yhat, rexp, tp = g.decompose(p, xhat, mats)
    err_bps = (y - yhat) * 1e4
    pe = pd.DataFrame({
        "Mean (bps)": err_bps.mean(axis=0), "Std (bps)": err_bps.std(axis=0, ddof=1),
        "RMSE (bps)": np.sqrt((err_bps ** 2).mean(axis=0)), "Max abs (bps)": np.abs(err_bps).max(axis=0),
    }, index=[f"{m / 12:g} y" for m in mats])
    print("\nPricing errors (actual minus model, filtered states)")
    print(pe.to_string(float_format=lambda v: f"{v:10.2f}"))
    pe.to_csv(out / "pricing_errors.csv")

    dec = pd.DataFrame({
        "Fitted yield (%)": yhat.mean(axis=0) * 100, "Expected short rate (%)": rexp.mean(axis=0) * 100,
        "Term premium (%)": tp.mean(axis=0) * 100}, index=[f"{m / 12:g} y" for m in mats])
    print("\nTime averages of the yield decomposition")
    print(dec.to_string(float_format=lambda v: f"{v:10.3f}"))
    dec.to_csv(out / "decomposition_averages.csv")

    with open(out / "params_estimated.json", "w") as fh:
        json.dump({"params": g.params_to_dict(p), "std_errors": dict(zip(g.PARAM_NAMES, map(float, se))),
                   "avg_loglik": float(-best.nll / T), "n_obs": T, "yields_file": yields_path.name,
                   "maturities_years": [m / 12 for m in mats]}, fh, indent=2)

    pl.yield_fit_figure(dates, y, yhat, mats, out / "yield_fit.png")
    pl.decomposition_figure(dates, yhat, rexp, tp, mats, out / "yield_decomposition.png")
    print(f"\nSaved estimates, tables and figures in {out}/")


if __name__ == "__main__":
    main()