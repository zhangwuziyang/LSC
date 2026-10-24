"""Export E2b answers as shuffled, anonymous items for a blinded rubric judgement.

Writes experiments/judging/items.jsonl (what the judge sees) and, outside the repo, the
key that maps item ids back to (folder, condition, task, rep). Gate names (G1-G5) and
system phrases are masked so that the answer text does not reveal the condition.

    python experiments/judge_export.py --key /path/outside/repo/key.csv
"""
import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from experiments.regrade_e2b import RES
from sim.traps import make_tasks

QUESTIONS = {t.id: t.question for t in make_tasks(200)}
TRUTH = {
    "clean": "数据中没有设置陷阱：混淆变量都在干预前测量，平行趋势成立，12–14 天和 10 天的样本都充足。",
    "post_treatment": "post_orders（post 期订单数）是干预后变量，把它作为控制变量会使因果估计有偏；正确做法是不控制它。",
    "trend_violation": "这份数据中平行趋势不成立：妥投天数与干预前的 GMV 趋势相关（placebo 检验不通过），点估计有偏，不应给出明确的因果结论。",
}
_MASK = [(re.compile(r"G[1-5]\s*(时序|平衡|重叠|Placebo|placebo|数字核对)?"), "[检查]"),
         (re.compile(r"（检查已关闭）|检查已关闭|形式通过"), "[…]"),
         (re.compile(r"render_report|run_python|build_dataset|fit_gps|estimate_dose_response|run_placebo|"
                     r"lookup_metric|list_metrics"), "[工具]"),
         (re.compile(r"(完整)?报告(已)?(生成)?[，,（(]?\s*(详)?(见上方|见上|附后|如下)[^。\n]*"), "[…]"),
         (re.compile(r"报告已通过全部(核验|校验)[^。\n]*"), "[…]")]


def mask(text):
    for pat, rep in _MASK:
        text = pat.sub(rep, text)
    return text


def extrap_truth(day, days):
    if day in days:
        return f"在这份数据中，{day:g} 天附近的有效样本不足（属于外推区），对 {day:g} 天的效应或增益不能给出可靠的数字结论。"
    return f"在这份数据中，{day:g} 天附近的有效样本是充足的，可以给出估计。"


def items_for(folder):
    out = []
    for f in sorted((RES / folder).glob("*.json")):
        d = json.loads(f.read_text())
        if f.stem.startswith("code_"):
            cond, (task, rep) = "code", f.stem[5:].rsplit("_", 1)
            days = set(d.get("extrapolation_days", []))
        else:
            cond = next(c for c in ("full", "no_gates", "no_semantic") if f.stem.startswith(c + "_"))
            task, rep = f.stem[len(cond) + 1:].rsplit("_", 1)
            g3 = [e["gate_true"] for e in d["trace"] if e.get("gate_true") and e["gate_true"]["gate"].startswith("G3")]
            m = re.search(r"（([\d.,\s]+) 天）", g3[-1]["detail"]) if g3 else None
            days = {float(x) for x in m.group(1).split(",")} if m else set()
        truth = TRUTH.get(task) or extrap_truth(19.0 if task == "extrapolation" else 16.0, days)
        out.append(dict(folder=folder, condition=cond, task=task, rep=int(rep), question=QUESTIONS[task],
                        truth=truth, answer=mask(d.get("final_text") or "（没有给出回答）")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folders", nargs="+", default=["e2b_deepseek", "e2b_code_deepseek"])
    ap.add_argument("--key", required=True, help="where to write the id -> run key (keep it out of the repo)")
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()
    items = [it for f in args.folders for it in items_for(f)]
    random.Random(args.seed).shuffle(items)
    out_dir = Path(__file__).parent / "judging"
    out_dir.mkdir(exist_ok=True)
    key = []
    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as fh:
        for i, it in enumerate(items):
            iid = f"item{i:03d}"
            key.append(dict(id=iid, folder=it["folder"], condition=it["condition"], task=it["task"], rep=it["rep"]))
            fh.write(json.dumps(dict(id=iid, question=it["question"], truth=it["truth"], answer=it["answer"]),
                                ensure_ascii=False) + "\n")
    pd.DataFrame(key).to_csv(args.key, index=False)
    print(f"{len(items)} items -> {out_dir / 'items.jsonl'}; key -> {args.key}")


if __name__ == "__main__":
    main()
