"""Export E2 v2 answers as shuffled anonymous items for the pre-registered judging.

    python -m experiments.judge_export_v2 --key /path/outside/repo/key_v2.csv
"""
import argparse
import json
import random
from pathlib import Path

import pandas as pd

from experiments.judge_export import mask
from sim.traps_v2 import BY_ID, truth

ROOT = Path(__file__).parent / "results" / "e2v2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--seed", type=int, default=20261027)
    args = ap.parse_args()
    items, skipped = [], []
    for folder in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "reference"):
        for f in sorted(folder.glob("*.json")):
            rec = json.loads(f.read_text())
            if rec.get("error"):  # API failures are rerun, never judged
                skipped.append(f"{folder.name}/{f.name}")
                continue
            task = BY_ID[rec["task"]]
            ref = json.loads((ROOT / "reference" / f"{rec['task']}_{rec['rep']}.json").read_text())
            items.append(dict(folder=folder.name, task=rec["task"], rep=rec["rep"], question=task.question,
                              truth=truth(task, ref), answer=mask(rec.get("final_text") or "（没有给出回答）")))
    random.Random(args.seed).shuffle(items)
    out = Path(__file__).parent / "judging_v2"
    key = []
    with open(out / "items.jsonl", "w", encoding="utf-8") as fh:
        for i, it in enumerate(items):
            iid = f"v2item{i:04d}"
            key.append(dict(id=iid, folder=it["folder"], task=it["task"], rep=it["rep"]))
            fh.write(json.dumps(dict(id=iid, question=it["question"], truth=it["truth"], answer=it["answer"]),
                                ensure_ascii=False) + "\n")
    pd.DataFrame(key).to_csv(args.key, index=False)
    print(f"{len(items)} items")
    print(pd.DataFrame(key).groupby("folder").size().to_string())
    if skipped:
        print(f"skipped {len(skipped)} runs with API errors: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
