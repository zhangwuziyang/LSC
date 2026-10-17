"""Figures for E1 from experiments/results/e1_curves.csv.

    python experiments/plot_e1.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).parent / "results"
# Categorical slots 1-4 of the validated reference palette, in fixed order.
COLORS = {"E0+M0 (original)": "#eb6834", "E0+M1": "#eda100",
          "E1+M0": "#1baf7a", "E1+M1 (proposed)": "#2a78d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    cur = pd.read_csv(RES / "e1_curves.csv")
    cur["covered"] = (cur.lo <= cur.truth) & (cur.truth <= cur.hi)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelcolor": INK})
    for scen, df in cur.groupby("scenario", sort=False):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        mean = df.groupby(["method", "d"], sort=False)[["est", "truth", "covered"]].mean().reset_index()

        ax = axes[0]
        tr = mean[mean.method == mean.method.iloc[0]]
        ax.plot(tr.d, tr.truth, color=INK, lw=2, ls="--", label="truth")
        for m, c in COLORS.items():
            g = mean[mean.method == m]
            ax.plot(g.d, g.est, color=c, lw=2, label=m)
        ax.set(title="(a) mean estimate of θ(10, d)", xlabel="delivery days d", ylabel="Δ daily GMV")

        ax = axes[1]
        ax.axhline(0.95, color=MUTED, lw=1, ls=":")
        for m, c in COLORS.items():
            g = mean[(mean.method == m) & (mean.d != 10)]
            ax.plot(g.d, g.covered, color=c, lw=2, marker="o", ms=3)
        ax.set(title="(b) pointwise 95% CI coverage", xlabel="delivery days d", ylim=(0, 1.02))

        ax = axes[2]
        g = df[(df.method == "E1+M1 (proposed)") & (df.d != 10) & df.ess.notna()].copy()
        bins = np.array([0, 50, 100, 200, 500, 1000, 2000, np.inf])
        g["bin"] = pd.cut(g.ess, bins, right=False)
        cal = g.groupby("bin", observed=True).agg(cov=("covered", "mean"), n=("covered", "size"),
                                                   ess=("ess", "median"))
        ax.axhline(0.95, color=MUTED, lw=1, ls=":")
        ax.plot(cal.ess, cal["cov"], color=COLORS["E1+M1 (proposed)"], lw=2, marker="o", ms=5)
        ax.set_xscale("log")
        ax.set(title="(c) coverage vs ESS, E1+M1", xlabel="ESS at dose (log)", ylim=(0, 1.02))

        for a in axes:
            style(a)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=8)
        fig.suptitle(f"E1 · scenario: {scen}", color=INK)
        for ext in ("png", "pdf"):
            fig.savefig(RES / f"e1_{scen}.{ext}", dpi=160)
        plt.close(fig)
        cal.to_csv(RES / f"e1_ess_calibration_{scen}.csv")
        print(scen, "\n", cal.round(3))


if __name__ == "__main__":
    main()
