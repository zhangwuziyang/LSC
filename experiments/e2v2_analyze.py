"""E2 v2 analysis: pass rates by condition and task, judge agreement, and the paired comparisons
the pre-registration names (full vs fixed = value of the LLM; fixed vs code = value of engine
and gates; full vs no_gates = value of showing the gates).

    python -m experiments.e2v2_analyze --key experiments/judging_v2/key_v2.csv \
        --judges experiments/judging_v2/judgments_A.csv experiments/judging_v2/judgments_B.csv \
        [--arbiter experiments/judging_v2/judgments_C.csv]

Judgment files have columns id, verdict (pass/fail), reason.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from sim.traps_v2 import BY_ID

ROOT = Path(__file__).parent / "results" / "e2v2"
OUT = ROOT / "analysis"


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == v).mean() * (b == v).mean() for v in ("pass", "fail"))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def mcnemar(x, y):
    """Exact McNemar test on paired pass indicators."""
    b, c = int((x & ~y).sum()), int((~x & y).sum())
    p = binomtest(b, b + c).pvalue if b + c else 1.0
    return b, c, p


def run_costs():
    rows = []
    for folder in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name not in ("reference", "analysis")):
        for f in folder.glob("*.json"):
            r = json.loads(f.read_text())
            rows.append(dict(folder=folder.name, task=r["task"], rep=r["rep"], seconds=r.get("seconds"),
                             input_tokens=r.get("input_tokens"), output_tokens=r.get("output_tokens")))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--judges", nargs=2, required=True, help="judgment files of judges A and B")
    ap.add_argument("--arbiter", help="judgments of judge C on the A/B disagreements")
    args = ap.parse_args()

    key = pd.read_csv(args.key)
    A, B = (pd.read_csv(p).set_index("id")["verdict"].str.strip().str.lower() for p in args.judges)
    df = key.assign(A=key.id.map(A), B=key.id.map(B))
    missing = df[df.A.isna() | df.B.isna()]
    if len(missing):
        raise SystemExit(f"{len(missing)} items lack a verdict from A or B, e.g. {missing.id.head().tolist()}")
    df["majority"] = df.A.where(df.A == df.B)
    if args.arbiter:
        C = pd.read_csv(args.arbiter).set_index("id")["verdict"].str.strip().str.lower()
        df["majority"] = df.majority.fillna(df.id.map(C))
    df["category"] = df.task.map(lambda t: BY_ID[t].category)
    df["trap"] = df.category != "clean"
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "verdicts.csv", index=False)

    lines = ["# E2 v2 results", "",
             f"{len(df)} answers; A/B agreement {(df.A == df.B).mean():.1%}, Cohen's kappa {kappa(df.A, df.B):.2f}; "
             f"{(df.A != df.B).sum()} disagreements" + ("" if args.arbiter else " (no arbiter yet: majority column is A∩B only)"),
             ""]
    for col in ("A", "majority"):
        p = df.assign(ok=df[col] == "pass")
        lines += [f"## Passes by condition and task ({col})", "",
                  p.pivot_table(index="task", columns="folder", values="ok", aggfunc="sum")
                   .reindex([t for t in BY_ID if t in set(df.task)]).to_markdown(), "",
                  p.groupby(["folder", "trap"]).ok.agg(["sum", "count"]).unstack().to_markdown(), ""]

    # Paired comparisons on the same dataset (task, rep). The fixed workflow has no provider.
    ok = df.assign(ok=df.A == "pass").pivot_table(index=["task", "rep"], columns="folder", values="ok", aggfunc="first")
    lines += ["## Paired comparisons (judge A; exact McNemar; b = only first passes, c = only second passes)", "",
              "| comparison | n pairs | first | second | b | c | p |", "|---|---|---|---|---|---|---|"]
    providers = sorted({f.rsplit("_", 1)[1] for f in ok.columns if f != "fixed"})
    pairs = []
    for prov in providers:
        pairs += [(f"full_{prov}", "fixed"), ("fixed", f"code_{prov}"), (f"full_{prov}", f"no_gates_{prov}"),
                  (f"full_{prov}", f"code_{prov}")]
    for x, y in pairs:
        if x not in ok or y not in ok:
            continue
        sub = ok[[x, y]].dropna().astype(bool)
        b, c, pv = mcnemar(sub[x].to_numpy(), sub[y].to_numpy())
        lines.append(f"| {x} vs {y} | {len(sub)} | {sub[x].mean():.0%} | {sub[y].mean():.0%} | {b} | {c} | {pv:.3g} |")

    costs = run_costs()
    if len(costs):
        lines += ["", "## Cost per run (median)", "",
                  costs.groupby("folder")[["seconds", "input_tokens", "output_tokens"]].median().round(0).to_markdown()]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
