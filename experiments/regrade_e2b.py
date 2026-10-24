"""Regrade saved E2b runs with the current task checks, from their JSON traces.

The checks only read the final answer, the report and the gates' true verdicts, all of
which are saved; the G5 verdict itself cannot be recomputed offline (the engine numbers
are not saved), so runs whose only failure is all_numbers_verified are flagged.

    python experiments/regrade_e2b.py e2b_deepseek e2b_qwen
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd

from agent.gates import GateResult
from sim.traps import make_tasks

RES = Path(__file__).parent / "results"


def run_view(d):
    gates, covs = {}, None
    for e in d["trace"]:
        g = e.get("gate_true") or e.get("gate")
        if g:
            gates[g["gate"]] = GateResult(g["gate"], g["status"], g["detail"])
        if e["tool"] == "build_dataset" and not e["is_error"]:
            shown = e.get("gate") or {}
            if shown.get("status") != "block":
                covs = e["input"]["covariates"]
    g4 = gates.get("G4 Placebo")
    session = NS(covariates=covs, report=d.get("report"), gates=gates,
                 pl=None if g4 is None else {"passed": g4.status == "pass"})
    return NS(final_text=d.get("final_text") or "", trace=d["trace"], session=session)


def main(folders):
    checks = {t.id: t.checks for t in make_tasks(200)}  # data size is irrelevant for grading
    rows = []
    for folder in folders:
        for f in sorted((RES / folder).glob("*.json")):
            cond, task, rep = f.stem.rsplit("_", 2)[0], f.stem.rsplit("_", 2)[1], int(f.stem.rsplit("_", 1)[1])
            for c in ("full", "no_gates", "no_semantic"):
                if f.stem.startswith(c + "_"):
                    cond, task = c, f.stem[len(c) + 1:].rsplit("_", 1)[0]
            res = checks[task](run_view(json.loads(f.read_text())))
            failed = [k for k, v in res.items() if not v]
            rows.append(dict(folder=folder, condition=cond, task=task, rep=rep, passed=not failed,
                             failed=",".join(failed), g5_only=failed == ["all_numbers_verified"]))
    df = pd.DataFrame(rows)
    out = RES / "e2b_regraded.csv"
    df.to_csv(out, index=False)
    print(df.groupby(["folder", "condition"]).agg(runs=("passed", "size"), passed=("passed", "sum"),
                                                  g5_only=("g5_only", "sum")).to_string())
    print(df[~df.passed].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["e2b_deepseek", "e2b_qwen"])
