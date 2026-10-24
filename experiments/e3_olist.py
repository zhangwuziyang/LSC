"""E3: Olist case study with the gated pipeline (no LLM).

    git clone --depth 1 https://github.com/olist/work-at-olist-data ../olist-data
    python experiments/e3_olist.py --data ../olist-data/datasets
"""
import argparse
import json
from pathlib import Path

import numpy as np

from agent.pipeline import run_analysis
from gpsdid import NormalGPS, hirano_imbens
from agent.report import render
from agent.semantic import SemanticLayer
from cases.olist import COVARIATES, build_panel

OUT = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--min-orders", type=int, default=3)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    df = build_panel(args.data, min_orders=args.min_orders)
    print(f"panel: {len(df)} seller-months, {df.seller_id.nunique()} sellers")
    sem = SemanticLayer(Path(__file__).parents[1] / "agent" / "semantic_layer" / "olist_metrics.yaml")
    result = run_analysis(df, semantic=sem, covariates=COVARIATES, grid=np.arange(6.0, 25.01, 0.5),
                          d0=10.0, days=1, kink_range=(7, 22), from_days=(14, 16, 18, 20), to_days=(10, 12),
                          estimator_kw=dict(gps="auto"))
    text, g5 = render(result)
    # The industry baseline on the same data: normal GPS + Hirano-Imbens, iid bootstrap.
    grid = np.arange(6.0, 25.01, 0.5)
    hi = hirano_imbens(df[COVARIATES].to_numpy(float), df.D.to_numpy(), df.dY.to_numpy(), grid, gps=NormalGPS)
    p, dr = hi.contrast(10.0)
    lo, up = np.quantile(dr, [0.025, 0.975], axis=0)
    rows = [f"| {d:g} | {p[i]:.3f} | [{lo[i]:.3f}, {up[i]:.3f}] |" for i, d in enumerate(grid) if d % 2 == 0]
    text += ("\n\n## 对照：原业务做法（正态 GPS + 两步法，不分卖家的 bootstrap）\n\n"
             "| 妥投天数 | θ | 95% CI |\n|---|---|---|\n" + "\n".join(rows) + "\n")
    tag = args.tag or f"min{args.min_orders}"
    (OUT / f"e3_olist_report_{tag}.md").write_text(
        f"# Olist 案例（{len(df)} 个卖家×月，{df.seller_id.nunique()} 个卖家）\n\n" + text, encoding="utf-8")
    (OUT / f"e3_olist_result_{tag}.json").write_text(json.dumps(
        {k: v for k, v in result.items() if k != "gates"} | {"gates": [g.to_dict() for g in result["gates"]]},
        ensure_ascii=False, indent=1, default=float))
    print(text)


if __name__ == "__main__":
    main()
