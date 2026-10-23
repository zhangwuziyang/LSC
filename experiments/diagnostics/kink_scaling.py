"""How does the kink estimate sharpen with n? Proposed method only, heavy-tail scenario.
Writes experiments/results/diag_kink_scaling.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from gpsdid import dr_dose_response, kink
from sim.dgp import simulate

GRID = np.arange(5.0, 20.01, 0.5)
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
SIZES = [20000, 100000]

rows = []
for n in SIZES:
    for rep in range(REPS):
        s = simulate(n, seed=200 + rep)
        r = dr_dose_response(s.X, s.D, s.dY, GRID, seed=rep, n_boot=200)
        k, (lo, hi) = kink(r, 6, 18, seed=rep)
        rows.append(dict(n=n, rep=rep, kink=k, lo=lo, hi=hi, covers=lo <= 10 <= hi))
        print(f"n={n} rep {rep + 1}/{REPS}: kink={k:.2f} [{lo:.2f}, {hi:.2f}]", flush=True)
        pd.DataFrame(rows).to_csv(Path(__file__).parents[1] / "results" / "diag_kink_scaling.csv", index=False)

df = pd.DataFrame(rows)
print(df.assign(err=(df.kink - 10).abs(), width=df.hi - df.lo)
        .groupby("n")[["err", "width", "covers"]].mean().round(3))
