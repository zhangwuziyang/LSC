"""Precompute the reference facts (extrapolation doses, kink CI, placebo verdict) for every
E2 v2 dataset, so all conditions and machines grade against the same facts.

    python -m experiments.e2v2_reference --reps 3
"""
import argparse
import json
from pathlib import Path

from sim.traps_v2 import TASKS, make_df, reference

OUT = Path(__file__).parent / "results" / "e2v2" / "reference"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        for rep in range(args.reps):
            f = OUT / f"{task.id}_{rep}.json"
            if f.exists():
                continue
            ref = reference(task, make_df(task, rep))
            f.write_text(json.dumps(ref, default=float))
            print(task.id, rep, ref, flush=True)


if __name__ == "__main__":
    main()
