"""What weighted |corr(X, D)| does G2 see under a good (M1) and a misspecified (M0) GPS,
and does it predict estimation error? Writes experiments/results/diag_balance.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gpsdid import HeteroGBMGPS, NormalGPS, balance_table, dr_dose_response
from sim.dgp import COVARIATES, simulate

GRID = np.arange(6.0, 16.01, 0.5)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
rows = []
for rep in range(REPS):
    s = simulate(20000, seed=400 + rep)
    truth = s.true_contrast(10, GRID)
    for name, gps in [("M1", HeteroGBMGPS), ("M0", NormalGPS)]:
        r = dr_dose_response(s.X, s.D, s.dY, GRID, gps=gps, seed=rep, n_boot=100, inflate=False)
        tbl = balance_table(s.X, s.D, r.weights, COVARIATES)
        clipped = np.minimum(r.weights, np.quantile(r.weights, 0.999))
        tbl_c = balance_table(s.X, s.D, clipped, COVARIATES)
        p, _ = r.contrast(10)
        rows.append(dict(rep=rep, gps=name, max_corr=tbl.corr_weighted.abs().max(),
                         max_corr_clipped=tbl_c.corr_weighted.abs().max(),
                         worst=tbl.covariate[tbl.corr_weighted.abs().idxmax()],
                         rmse=np.sqrt(np.mean((p - truth)[GRID != 10] ** 2))))
    print(rows[-2], rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(Path(__file__).parents[1] / "results" / "diag_balance.csv", index=False)
df = pd.DataFrame(rows)
print(df.groupby("gps")[["max_corr", "max_corr_clipped", "rmse"]].describe().round(3).T)
