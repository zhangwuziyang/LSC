"""Figure for E2a: G4 rejection rate and placebo deviation vs. main-estimate bias by tau."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).parent / "results"
INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"

df = pd.concat([pd.read_csv(f) for f in sorted(RES.glob("e2a_placebo_taus*.csv"))])
g = df.groupby("tau").agg(reject=("g4_pass", lambda x: 1 - x.mean()), n=("g4_pass", "size"),
                          placebo=("placebo_max_dev", "mean"), bias=("main_max_bias", "mean")).reset_index()
se = np.sqrt(g.reject * (1 - g.reject) / g.n)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
ax = axes[0]
ax.errorbar(g.tau, g.reject, yerr=1.96 * se, color=BLUE, lw=2, marker="o", ms=6, capsize=3)
ax.axhline(0.05, color=MUTED, lw=1, ls=":")
ax.set(title="(a) G4 rejection rate", xlabel="trend violation τ", ylabel="share of runs flagged", ylim=(-0.05, 1.05))
ax = axes[1]
ax.plot(g.tau, g.bias, color=ORANGE, lw=2, marker="o", ms=6, label="max |bias| of main estimate")
ax.plot(g.tau, g.placebo, color=BLUE, lw=2, marker="o", ms=6, label="max |placebo deviation|")
ax.set(title="(b) placebo deviation tracks bias", xlabel="trend violation τ", ylabel="daily GMV")
ax.legend(frameon=False, fontsize=8)
for a in axes:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.tick_params(colors=MUTED, labelsize=8)
    a.grid(color=GRID, lw=0.6)
for ext in ("png", "pdf"):
    fig.savefig(RES / f"e2a_placebo_gate.{ext}", dpi=160)
print(g.round(3).to_string(index=False))
