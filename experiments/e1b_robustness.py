"""E1b: estimator robustness beyond the GPS's own data-generating form.

Scenarios: a bimodal dose (two shipping channels), a gamma dose with covariate-dependent
skewness, strong nonlinear confounding, and small / large samples. Same target and
metrics as E1.

    python experiments/e1b_robustness.py --scenario bimodal --reps 15
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e1_estimators import D0, GRID, evaluate
from gpsdid import HeteroGBMGPS, NormalGPS, dr_dose_response, hirano_imbens
from sim.dgp import simulate

SCENARIOS = {
    "bimodal": dict(n=20000, dose_model="bimodal"),
    "gamma": dict(n=20000, dose_model="gamma"),
    "strong_confounding": dict(n=20000, confounding=2.5, interaction=1.0),
    "n5k": dict(n=5000),
    "n50k": dict(n=50000),
}
METHODS = {
    "E0+M0 (original)": lambda s, seed: hirano_imbens(s.X, s.D, s.dY, GRID, gps=NormalGPS, seed=seed),
    "E1+M0": lambda s, seed: dr_dose_response(s.X, s.D, s.dY, GRID, gps=NormalGPS, seed=seed),
    "E1+M1": lambda s, seed: dr_dose_response(s.X, s.D, s.dY, GRID, gps=HeteroGBMGPS, seed=seed),
    "E1+M1 auto (proposed)": lambda s, seed: dr_dose_response(s.X, s.D, s.dY, GRID, gps="auto", seed=seed),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), required=True)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--start", type=int, default=0, help="first replication index (to split long runs)")
    args = ap.parse_args()
    out = Path(__file__).parent / "results" / f"e1b_{args.scenario}.csv"
    prev = pd.read_csv(out) if out.exists() else pd.DataFrame()
    rows = []
    for rep in range(args.start, args.start + args.reps):
        s = simulate(seed=1000 + rep, **SCENARIOS[args.scenario])
        truth = s.true_contrast(D0, GRID)
        for name, fn in METHODS.items():
            t = time.time()
            met, _ = evaluate(fn(s, rep), truth)
            rows.append(dict(scenario=args.scenario, rep=rep, method=name, seconds=time.time() - t, **met))
        print(f"{args.scenario} rep {rep} done", flush=True)
        pd.concat([prev, pd.DataFrame(rows)]).drop_duplicates(["scenario", "rep", "method"], keep="last") \
          .to_csv(out, index=False)
    df = pd.read_csv(out)
    print(df.groupby("method", sort=False)[["rmse", "abs_bias", "coverage", "band_covers", "ci_width",
                                            "kink_err", "seconds"]].mean().round(3))


if __name__ == "__main__":
    main()
