"""Does subsampling variance inflation restore 95% coverage? Heavy-tail scenario,
proposed method. Writes experiments/results/diag_coverage_fix.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gpsdid import dr_dose_response
from sim.dgp import simulate

GRID = np.arange(6.0, 16.01, 0.5)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
N_SUB = int(sys.argv[3]) if len(sys.argv) > 3 else 12
rows = []
for rep in range(REPS):
    s = simulate(20000, seed=300 + rep)
    truth = s.true_contrast(10, GRID)
    r = dr_dose_response(s.X, s.D, s.dY, GRID, seed=rep, n_boot=400, inflate=True,
                          sub_frac=FRAC, n_sub=N_SUB)
    p, dr = r.contrast(10)
    raw = p + (dr - p) / np.sqrt(r.vif)
    for name, d in [("multiplier", raw), ("inflated", dr)]:
        lo, hi = np.quantile(d, [0.025, 0.975], axis=0)
        rows.append(pd.DataFrame(dict(variant=name, frac=FRAC, rep=rep, vif=r.vif, d=GRID, est=p, truth=truth,
                                      lo=lo, hi=hi)))
    print(f"rep {rep + 1}/{REPS} vif={r.vif:.3f}", flush=True)
    out = pd.concat(rows)
    out.to_csv(Path(__file__).parents[1] / "results" / f"diag_coverage_fix_frac{FRAC}.csv", index=False)

e = out[out.d != 10]
e = e.assign(cov=(e.lo <= e.truth) & (e.truth <= e.hi), width=e.hi - e.lo)
print(e.groupby("variant")[["cov", "width", "vif"]].mean().round(3))
