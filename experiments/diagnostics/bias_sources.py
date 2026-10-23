"""Which nuisance choice drives the DR bias? Compares weight clipping and outcome-model
capacity on the heavy-tail scenario. Writes experiments/results/diag_bias_sources.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from gpsdid import dr_dose_response
from sim.dgp import simulate

GRID = np.arange(6.0, 16.01, 0.5)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def big_outcome(seed=0):
    return LGBMRegressor(n_estimators=800, learning_rate=0.03, num_leaves=31,
                         min_child_samples=20, random_state=seed, verbose=-1)


VARIANTS = {
    "base": {},
    "no_clip": dict(weight_clip=1.0),
    "big_outcome": dict(outcome=big_outcome),
    "no_clip+big_outcome": dict(weight_clip=1.0, outcome=big_outcome),
}

rows = []
for rep in range(REPS):
    s = simulate(20000, seed=100 + rep)
    truth = s.true_contrast(10, GRID)
    for name, kw in VARIANTS.items():
        r = dr_dose_response(s.X, s.D, s.dY, GRID, seed=rep, n_boot=300, **kw)
        p, dr = r.contrast(10)
        rows.append(pd.DataFrame(dict(variant=name, rep=rep, d=GRID, est=p, truth=truth, se=dr.std(0))))
    print(f"rep {rep + 1}/{REPS}", flush=True)
    out = pd.concat(rows)
    out.to_csv(Path(__file__).parents[1] / "results" / "diag_bias_sources.csv", index=False)

e = out[out.d != 10]
summ = e.assign(err=e.est - e.truth).groupby("variant").apply(
    lambda g: pd.Series(dict(mean_bias=g.err.mean(), rmse=np.sqrt((g.err ** 2).mean()),
                             sd_over_se=g.groupby("d").est.std().mean() / g.se.mean(),
                             coverage=(np.abs(g.err) <= 1.96 * g.se).mean())), include_groups=False)
print(summ.round(4))
