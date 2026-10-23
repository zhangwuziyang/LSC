"""E2a: operating characteristics of the placebo gate G4.

For each trend-violation strength tau, estimate (i) the placebo curve and the G4 verdict
and (ii) the bias of the main estimate, which the same violation induces. With tau = 0
the rejection rate is G4's false-alarm rate; with tau > 0 it is its power.

    python experiments/e2a_placebo_gate.py --taus 0 0.25 --reps 15
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gpsdid import dr_dose_response, grade_overlap, placebo_check
from sim.dgp import simulate

GRID = np.arange(5.0, 20.01, 0.5)
D0 = 10.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()
    out = Path(__file__).parent / "results" / f"e2a_placebo_taus{'_'.join(f'{t:g}' for t in args.taus)}.csv"
    rows = []
    for tau in args.taus:
        for rep in range(args.reps):
            s = simulate(args.n, tau=tau, seed=500 + rep)
            truth = s.true_contrast(D0, GRID)
            main_res = dr_dose_response(s.X, s.D, s.dY, GRID, seed=rep)
            usable = grade_overlap(main_res.ess) != "extrapolation"
            pl = dr_dose_response(s.X, s.D, s.dY_placebo, GRID, seed=rep)
            passed, pl_p, *_ = placebo_check(pl, D0, mask=usable)
            p, _ = main_res.contrast(D0)
            m = usable & (GRID != D0)
            rows.append(dict(tau=tau, rep=rep, g4_pass=passed,
                             placebo_max_dev=np.abs(pl_p[m]).max(),
                             main_max_bias=np.abs((p - truth)[m]).max(),
                             main_rmse=np.sqrt(np.mean((p - truth)[m] ** 2))))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    print(df.groupby("tau").agg(reject_rate=("g4_pass", lambda x: 1 - x.mean()),
                                placebo_dev=("placebo_max_dev", "mean"),
                                main_bias=("main_max_bias", "mean"),
                                main_rmse=("main_rmse", "mean")).round(3))


if __name__ == "__main__":
    main()
