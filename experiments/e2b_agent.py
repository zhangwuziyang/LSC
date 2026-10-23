"""E2b: the agent on the trap benchmark, full system vs ablations.

Needs API credentials: ANTHROPIC_API_KEY for Claude (default), DEEPSEEK_API_KEY with
--provider deepseek, or DASHSCOPE_API_KEY with --provider qwen (environment settings or
.env; the network policy must allow the provider's host). Check with
`python -m agent.check_providers` first.
Each run costs real tokens; start with --reps 1 --tasks clean to check the setup.

    python experiments/e2b_agent.py --reps 3
    python experiments/e2b_agent.py --conditions full no_gates --tasks post_treatment --reps 1
    python experiments/e2b_agent.py --provider deepseek --tasks clean --reps 1
    python experiments/e2b_agent.py --provider qwen --reps 3
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd

from agent.orchestrator import run_agent
from sim.traps import make_tasks

CONDITIONS = {
    "full": dict(),
    "no_gates": dict(gates_enabled=False),
    "no_semantic": dict(semantic_enabled=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--effort", default="high", help="Claude only")
    ap.add_argument("--provider", choices=["anthropic", "deepseek", "qwen"], default="anthropic")
    ap.add_argument("--model", default=None, help="override the provider's default model")
    args = ap.parse_args()
    out_dir = Path(__file__).parent / "results" / f"e2b_{args.provider}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [t for t in make_tasks(args.n) if args.tasks is None or t.id in args.tasks]
    rows = []
    for cond in args.conditions:
        for task in tasks:
            for rep in range(args.reps):
                t0 = time.time()
                run = run_agent(task.df, task.question, session_kw=dict(seed=rep, **CONDITIONS[cond]),
                                effort=args.effort, provider=args.provider, model=args.model)
                checks = task.checks(run)
                rows.append(dict(provider=args.provider, condition=cond, task=task.id, rep=rep, passed_all=all(checks.values()),
                                 turns=run.turns, tool_calls=len(run.trace), seconds=time.time() - t0,
                                 stop_reason=run.stop_reason, **run.usage, **checks))
                stem = f"{cond}_{task.id}_{rep}"
                (out_dir / f"{stem}.json").write_text(json.dumps(
                    dict(final_text=run.final_text, trace=run.trace, checks=checks,
                         report=run.session.report), ensure_ascii=False, indent=1, default=str))
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    df = pd.DataFrame(rows)
    print(df.groupby(["condition", "task"])["passed_all"].mean().unstack().round(2))


if __name__ == "__main__":
    main()
