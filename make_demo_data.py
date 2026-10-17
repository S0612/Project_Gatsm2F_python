"""Write a SIMULATED yield data set in the same format as the real data files.

The thesis used monthly US yields, January 1972 to December 2025:
  <name>.csv              T x M matrix of yields in percent, no header
  <name>_maturities.csv   1 x M row of maturities in years

This script simulates such files from the model, so that the code can be run
without the real data. The numbers are simulated and mean nothing economically.
"""

import argparse
from pathlib import Path

import numpy as np

import gatsm as g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/demo_yields.csv")
    ap.add_argument("--months", type=int, default=648)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    years = np.arange(1, 16)                       # 1 to 15 years
    mats = years * 12
    p = g.params_from_dict(g.START_VALUES)
    y, _ = g.simulate(p, mats, args.months, np.random.default_rng(args.seed))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, y * 100, delimiter=",", fmt="%.5f")
    mats_path = out.with_name(out.stem + "_maturities.csv")
    np.savetxt(mats_path, years[None, :], delimiter=",", fmt="%d")
    print(f"wrote {out} and {mats_path}")


if __name__ == "__main__":
    main()
