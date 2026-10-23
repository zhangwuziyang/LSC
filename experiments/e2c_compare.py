"""E2c: compare providers on the E2b trap benchmark (reads experiments/results/e2b_*/summary.csv).

    python experiments/e2c_compare.py
"""
from pathlib import Path

import pandas as pd

RES = Path(__file__).parent / "results"
frames = [pd.read_csv(f) for f in sorted(RES.glob("e2b_*/summary.csv"))]
if not frames:
    raise SystemExit("no E2b results yet; run experiments/e2b_agent.py first")
df = pd.concat(frames)
by_task = df.pivot_table(index=["provider", "condition"], columns="task", values="passed_all", aggfunc="mean")
by_task["overall"] = df.groupby(["provider", "condition"]).passed_all.mean()
cost = df.groupby(["provider", "condition"]).agg(turns=("turns", "mean"), tool_calls=("tool_calls", "mean"),
                                                 input_tokens=("input_tokens", "mean"),
                                                 output_tokens=("output_tokens", "mean"),
                                                 minutes=("seconds", lambda s: s.mean() / 60))
checks = [c for c in df.columns if c in {"no_post_treatment_covariate", "all_numbers_verified",
                                         "finished_with_report", "explains_exclusion", "reports_caveat",
                                         "declines_extrapolation"}]
per_check = df.groupby(["provider", "condition"])[checks].mean()
text = ("# E2c: provider comparison\n\n## Share of tasks passed\n\n" + by_task.round(2).to_markdown() +
        "\n\n## Individual checks\n\n" + per_check.round(2).to_markdown() +
        "\n\n## Cost per run\n\n" + cost.round(1).to_markdown() + "\n")
(RES / "e2c_compare.md").write_text(text, encoding="utf-8")
print(text)
