"""Figure for E3 (Olist): gated DR estimate with CI and M=1 sensitivity band vs. the
industry estimator (normal GPS + Hirano-Imbens, iid bootstrap) on the same panel.

    python experiments/plot_e3.py --data ../olist-data/datasets
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cases.olist import COVARIATES, build_panel
from gpsdid import NormalGPS, hirano_imbens

RES = Path(__file__).parent / "results"
INK, MUTED, GRID, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    args = ap.parse_args()
    r = json.loads((RES / "e3_olist_result_min3.json").read_text())
    c = {k: np.array(v) if k != "grade" else v for k, v in r["curve"].items()}
    keep = np.array([g != "extrapolation" for g in c["grade"]])
    d = c["d"][keep]

    df = build_panel(args.data, min_orders=3)
    grid = np.arange(6.0, 25.01, 0.5)
    hi = hirano_imbens(df[COVARIATES].to_numpy(float), df.D.to_numpy(), df.dY.to_numpy(), grid, gps=NormalGPS)
    p, dr = hi.contrast(10.0)
    lo, up = np.quantile(dr, [0.025, 0.975], axis=0)

    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.fill_between(d, c["sens_lo"][keep], c["sens_hi"][keep], color=BLUE, alpha=0.10, lw=0,
                    label="ours: sensitivity interval (M = 1)")
    ax.fill_between(d, c["lo"][keep], c["hi"][keep], color=BLUE, alpha=0.25, lw=0, label="ours: 95% CI")
    ax.plot(d, c["theta"][keep], color=BLUE, lw=2, label="ours: estimate (placebo fails)")
    m = (grid >= d.min()) & (grid <= d.max())
    ax.fill_between(grid[m], lo[m], up[m], color=ORANGE, alpha=0.20, lw=0)
    ax.plot(grid[m], p[m], color=ORANGE, lw=2, label="industry baseline: estimate, 95% CI")
    ax.set(xlabel="seller's mean delivery days in month T", ylabel="θ(10, d), daily GMV (BRL)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    for ext in ("png", "pdf"):
        fig.savefig(RES / f"e3_olist.{ext}", dpi=160)


if __name__ == "__main__":
    main()
