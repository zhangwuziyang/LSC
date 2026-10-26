"""E2 v2 runner. Conditions: full (agent with gates), no_gates (verdicts hidden from the
model), code (LLM writes its own code), fixed (no LLM; engine + gates as a fixed workflow).

    python -m experiments.e2v2 --condition fixed --reps 3
    python -m experiments.e2v2 --condition full --provider deepseek --reps 3 --skip-done
"""
import argparse
import json
import time
from pathlib import Path

from agent.code_baseline import run_code_baseline
from agent.fixed_pipeline import run_fixed
from agent.orchestrator import run_agent
from agent.semantic import SemanticLayer
from sim.traps_v2 import BY_ID, TASKS, make_df

ROOT = Path(__file__).parent / "results" / "e2v2"
SEMANTIC = Path(__file__).parents[1] / "agent" / "semantic_layer" / "sim_metrics_v2.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["full", "no_gates", "code", "fixed"], required=True)
    ap.add_argument("--provider", choices=["deepseek", "qwen", "anthropic"], default=None)
    ap.add_argument("--tasks", nargs="+", default=[t.id for t in TASKS])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()
    if args.condition != "fixed" and args.provider is None:
        ap.error("--provider is required for LLM conditions")
    out = ROOT / (args.condition if args.condition == "fixed" else f"{args.condition}_{args.provider}")
    out.mkdir(parents=True, exist_ok=True)
    semantic = SemanticLayer(SEMANTIC)
    for tid in args.tasks:
        task = BY_ID[tid]
        for rep in range(args.reps):
            f = out / f"{tid}_{rep}.json"
            if args.skip_done and f.exists() and not json.loads(f.read_text()).get("error"):
                continue
            df = make_df(task, rep)
            t0 = time.time()
            rec = dict(task=tid, rep=rep, condition=args.condition, provider=args.provider)
            try:
                if args.condition == "fixed":
                    s, text = run_fixed(df, semantic)
                    rec.update(final_text=text, final_covariates=s.covariates,
                               gates={k: g.to_dict() for k, g in s.gates.items()})
                elif args.condition == "code":
                    run = run_code_baseline(df, task.question, provider=args.provider, semantic=semantic)
                    rec.update(final_text=run.final_text, trace=run.trace, turns=run.turns,
                               stop_reason=run.stop_reason, **run.usage)
                else:
                    kw = dict(semantic=semantic, seed=rep, gates_enabled=args.condition == "full")
                    run = run_agent(df, task.question, session_kw=kw, provider=args.provider)
                    rec.update(final_text=run.final_text, trace=run.trace, turns=run.turns,
                               stop_reason=run.stop_reason, final_covariates=run.session.covariates,
                               report=run.session.report, **run.usage)
            except Exception as e:  # record and continue; --skip-done reruns it
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["seconds"] = time.time() - t0
            f.write_text(json.dumps(rec, ensure_ascii=False, indent=1, default=str))
            print(tid, rep, rec.get("error") or f"{rec['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
