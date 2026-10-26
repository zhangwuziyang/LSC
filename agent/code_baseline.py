"""Baseline for E2b: the LLM answers the same question by writing and running its own
Python code. Same data, same data dictionary (including each column's measurement
period), same provider loop as the agent, but no engine, no gates and no number check.

Generated code runs in a fresh subprocess in a temporary directory, with a time limit and
an environment that holds no API keys.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .orchestrator import AgentRun, _loop_anthropic, _loop_openai_compat, make_client, provider_config
from .semantic import SemanticLayer

SYSTEM_CODE = """你是一名数据分析师，负责回答"物流妥投天数对用户 GMV 的影响"这类因果问题。
数据在当前目录的 data.csv。你可以用 run_python 工具执行 Python 代码（pandas、numpy、scipy、
scikit-learn、lightgbm、statsmodels 可用），代码的标准输出会返回给你；每次执行都是新的进程，需要重新读取数据。
你可以使用任何你认为合适的方法。你最多可以调用 {budget} 次工具，请合理安排；分析完成后不再调用工具，
直接用中文给出结论，包括具体数字及其不确定性。

数据字典（列名｜含义｜测量时期）：
{dictionary}"""

TOOL = [{"name": "run_python", "description": "执行一段 Python 代码，返回标准输出和标准错误（最多 4000 字符）。",
         "input_schema": {"type": "object", "properties": {"code": {"type": "string"}},
                          "required": ["code"], "additionalProperties": False}}]

TIMEOUT = 300
MAX_OUT = 4000


def data_dictionary(df, semantic):
    lines = []
    for c in df.columns:
        m = semantic.by_column.get(c)
        if m is None:
            continue
        desc = m.get("definition") or m["name"]
        lines.append(f"- {c}｜{m['name']}：{desc}｜{m.get('period') or '未标注'}")
    return "\n".join(lines)


class Sandbox:
    def __init__(self, df):
        self.dir = Path(tempfile.mkdtemp(prefix="code_baseline_"))
        df.to_csv(self.dir / "data.csv", index=False)
        self.env = {"PATH": os.environ.get("PATH", ""), "HOME": str(self.dir), "MPLBACKEND": "Agg",
                    "OMP_NUM_THREADS": "2"}

    def run(self, code):
        script = self.dir / "script.py"
        script.write_text(code, encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, "-I", str(script)], cwd=self.dir, env=self.env,
                               capture_output=True, text=True, timeout=TIMEOUT)
            out, ok = (p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr else "")), p.returncode == 0
        except subprocess.TimeoutExpired:
            out, ok = f"[timeout after {TIMEOUT}s]", False
        if len(out) > MAX_OUT:
            out = out[:MAX_OUT // 2] + "\n...[truncated]...\n" + out[-MAX_OUT // 2:]
        return out, ok


def run_code_baseline(df, question, provider="deepseek", client=None, model=None, semantic=None,
                      max_turns=30):
    semantic = semantic or SemanticLayer()
    data = df[[c for c in df.columns if c in semantic.by_column]]
    box = Sandbox(data)
    system = SYSTEM_CODE.format(dictionary=data_dictionary(data, semantic), budget=max_turns - 1)
    run = AgentRun(final_text="")
    run.code = []

    def execute(turn, name, args):
        if name != "run_python" or "code" not in args:
            return json.dumps({"error": "only run_python(code) is available"}), True
        out, ok = box.run(args["code"])
        run.code.append(args["code"])
        left = max_turns - 2 - turn
        if 0 < left <= 3:
            out += f"\n\n[提示] 你还剩 {left} 次工具调用，请准备给出结论。"
        run.trace.append({"turn": turn, "tool": name, "input": {"code": args["code"][:4000]},
                          "is_error": not ok, "output": out[:2000]})
        return out, not ok

    cfg = provider_config(provider)
    client = client or make_client(provider)
    if cfg["style"] == "anthropic":
        return _loop_anthropic(run, question, client, max_turns, model or cfg["model"], claude=False,
                               system=system, tools=TOOL, execute=execute, force_final=True)
    return _loop_openai_compat(run, question, client, max_turns, model or cfg["model"],
                               system=system, tools=TOOL, execute=execute, force_final=True)


# ---- grading ---------------------------------------------------------------------------
_EXCLUDES_POST = re.compile(r"(post_orders|post\s*期订单|干预后)[^。；\n]{0,40}(排除|剔除|删除|去掉|不纳入|未纳入|"
                            r"不控制|没有控制|未控制|不应|不能|不宜|不加入|没有加入)")


def grade_code_run(task_id, run, extrapolation_days=()):
    """Same intent as the agent checks in sim/traps.py, read from the answer and the code
    (there is no session or gate to read)."""
    from sim.traps import _declines_about, _reports_failed_placebo

    text = run.final_text or ""
    code = getattr(run, "code", [])
    out = {"finished_with_report": bool(re.search(r"\d", text))}
    if task_id == "post_treatment":
        # Pass if post_orders never entered the code, or if the answer says it is excluded
        # and does not name a post_orders model as the main estimate (one run did both:
        # "通常不应控制" and "以含 post_orders 的 DML 作为主估计").
        used = any("post_orders" in c for c in code)
        main_with_post = any(re.search(r"(主估计|首选|建议以|作为结论)", x) and re.search(r"(含|包括|加入)\s*post_orders", x)
                             for x in re.split(r"[。；\n]", text))
        out["no_post_treatment_covariate"] = (not used) or (bool(_EXCLUDES_POST.search(text)) and not main_with_post)
    elif task_id == "trend_violation":
        out["reports_caveat"] = _reports_failed_placebo(text)
    elif task_id == "extrapolation":
        out["declines_extrapolation"] = 19.0 not in extrapolation_days or _declines_about(text, 19.0)
    elif task_id == "thin_support":
        out["declines_extrapolation"] = 16.0 not in extrapolation_days or _declines_about(text, 16.0)
    return out
