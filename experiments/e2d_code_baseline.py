"""E2d: the code-writing baseline on the E2b trap tasks.

Grading needs, per task, the doses the engine grades as extrapolation (the agent's G3
verdict); they are computed once with the default estimator (seed 0).

    python experiments/e2d_code_baseline.py --provider deepseek --reps 3
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agent.code_baseline import grade_code_run, run_code_baseline
from gpsdid import dr_dose_response, grade_overlap
from sim.traps import PRE_COVS, make_tasks

GRID = np.arange(5.0, 20.01, 0.5)


def reference_extrapolation(df):
    res = dr_dose_response(df[PRE_COVS].to_numpy(float), df.D.to_numpy(), df.dY.to_numpy(), GRID,
                           inflate=False, n_boot=50, seed=0)
    return {float(d) for d, g in zip(GRID, grade_overlap(res.ess)) if g == "extrapolation"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["deepseek", "qwen"], default="deepseek")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--skip-done", action="store_true", help="skip (task, rep) runs already in summary.csv")
    args = ap.parse_args()
    out_dir = Path(__file__).parent / "results" / f"e2b_code_{args.provider}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = out_dir / "summary.csv"
    prev = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    done = set()
    if args.skip_done and len(prev):
        ok = prev[~prev.stop_reason.astype(str).str.startswith("error")]
        done = set(zip(ok.task, ok.rep))
    rows = []
    for task in [t for t in make_tasks(args.n) if args.tasks is None or t.id in args.tasks]:
        if all((task.id, r) in done for r in range(args.reps)):
            continue
        extrap = reference_extrapolation(task.df) if task.id in ("extrapolation", "thin_support") else set()
        for rep in range(args.reps):
            if (task.id, rep) in done:
                continue
            t0 = time.time()
            try:
                run = run_code_baseline(task.df, task.question, provider=args.provider, model=args.model)
            except Exception as e:
                print(f"{task.id}/{rep} failed: {type(e).__name__}: {e}", flush=True)
                rows.append(dict(provider=args.provider, condition="code", task=task.id, rep=rep,
                                 passed_all=False, stop_reason=f"error: {type(e).__name__}"))
                continue
            checks = grade_code_run(task.id, run, extrap)
            rows.append(dict(provider=args.provider, condition="code", task=task.id, rep=rep,
                             passed_all=all(checks.values()), turns=run.turns, tool_calls=len(run.trace),
                             seconds=time.time() - t0, stop_reason=run.stop_reason, **run.usage, **checks))
            (out_dir / f"code_{task.id}_{rep}.json").write_text(json.dumps(
                dict(final_text=run.final_text, trace=run.trace, checks=checks, extrapolation_days=sorted(extrap)),
                ensure_ascii=False, indent=1))
            print(rows[-1], flush=True)
            pd.concat([prev, pd.DataFrame(rows)]).drop_duplicates(["condition", "task", "rep"], keep="last") \
              .to_csv(summary, index=False)


if __name__ == "__main__":
    main()
