"""Run the gated pipeline on semi-synthetic data and write the report.

    python -m agent.run_demo --n 20000 --tau 0 --out experiments/results/demo_report.md
    python -m agent.run_demo --trap post_orders     # G1 should block
"""
import argparse
from pathlib import Path

import numpy as np

from sim.dgp import simulate

from .pipeline import run_analysis
from .report import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--trap", choices=["none", "post_orders"], default="none")
    ap.add_argument("--out", default="experiments/results/demo_report.md")
    args = ap.parse_args()

    s = simulate(args.n, tau=args.tau, seed=0)
    df = s.df.assign(post_orders=s.df.orders + np.random.default_rng(0).poisson(1, args.n))
    covs = None
    if args.trap == "post_orders":
        covs = ["log_hist_days", "region", "log_pre_gmv", "orders", "cat_share", "tenure", "post_orders"]
    else:
        covs = ["log_hist_days", "region", "log_pre_gmv", "orders", "cat_share", "tenure"]
    result = run_analysis(df, covariates=covs)
    text, g5 = render(result)
    if g5 is not None:
        text += f"\n\n---\n{g5.gate}：{g5.detail}\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
