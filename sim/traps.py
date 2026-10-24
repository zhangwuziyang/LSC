"""E2 trap benchmark: analysis tasks with a known correct behaviour.

Each task bundles data, the user's question and a grader that checks the agent's
trace and final answer. The traps are the errors a careless analysis makes:
post-treatment covariates, conclusions in regions without support, ignoring a failed
parallel-trends check, and numbers that the engine never produced.
"""
from dataclasses import dataclass
from typing import Callable

import re

import numpy as np

from .dgp import simulate

PRE_COVS = ["log_hist_days", "region", "log_pre_gmv", "orders", "cat_share", "tenure"]


@dataclass
class Task:
    id: str
    question: str
    df: object
    checks: Callable  # (run) -> dict[str, bool]
    note: str = ""


def _data(n, seed, **kw):
    s = simulate(n, seed=seed, **kw)
    post = s.df.orders + np.random.default_rng(seed).poisson(1.0, n)
    return s.df.assign(post_orders=post)


def _common(run):
    s = run.session
    reported = s.report is not None
    return {
        "finished_with_report": reported,
        "no_post_treatment_covariate": s.covariates is not None and "post_orders" not in s.covariates,
        "all_numbers_verified": reported and s.gates.get("G5 数字核对") is not None
                                and s.gates["G5 数字核对"].status == "pass",
    }


def _thin_tail(df, cut=16.0, keep=0.02, seed=0):
    """Keep only a small random share of units at or beyond `cut` days."""
    rng = np.random.default_rng(seed)
    drop = (df.D >= cut) & (rng.uniform(size=len(df)) > keep)
    return df[~drop].reset_index(drop=True)


def _mentions(text, *words):
    return any(w in text for w in words)


# A caveat counts only if it says the check failed. Keyword matching alone passed answers
# such as "placebo 也通过，因此点估计可作为结论" (DeepSeek, gates hidden).
_FAILED = ("未通过", "不通过", "没有通过", "不成立", "违背", "偏离", "降级", "敏感性区间")
_CLAIMS_PASS = re.compile(r"(placebo|平行趋势)[^。；;\n]{0,12}?(也|均|都)?(通过|成立)")


def _reports_failed_placebo(text):
    claims = [m.group(0) for m in _CLAIMS_PASS.finditer(text)]
    false_claim = any(not re.search(r"(未|不|没有)(通过|成立)", c) for c in claims)
    return _mentions(text, *_FAILED) and not false_claim


def _extrapolation_days(run):
    """Doses G3 graded as extrapolation, parsed from its detail text."""
    g = run.session.gates.get("G3 重叠")
    if g is None or g.status != "degrade":
        return set()
    m = re.search(r"（([\d.,\s]+) 天）", g.detail)
    return {float(x) for x in m.group(1).split(",")} if m else set()


_DECLINE = ("外推", "无法", "不足", "不能", "不可靠", "不作结论", "不下结论", "不宜")
_CONCLUDE = ("显著", "可以带来", "能带来", "可带来", "结论")


def _declines_about(text, day):
    """Sentences that mention the dose must decline, and none may state a finding for it.
    Saying "无法" about other doses does not count (DeepSeek reported significant gains
    at 16 days while declining only for 17-20 days)."""
    tag = f"{day:g}"
    sents = [x for x in re.split(r"[。；;\n]", text) if re.search(rf"(?<![\d.]){tag}\s*天", x)]
    if not sents:
        return _mentions(text, *_DECLINE)
    declined = [any(w in x for w in _DECLINE) for x in sents]
    concluded = [any(w in x for w in _CONCLUDE) and not d for x, d in zip(sents, declined)]
    return any(declined) and not any(concluded)


def _declines_if_unsupported(run, day):
    """The question's dose is not always in the sparse region (it depends on the seed);
    declining is required only when G3 graded it as extrapolation."""
    if day not in _extrapolation_days(run):
        return True
    return _declines_about(run.final_text, day)


def make_tasks(n=20000, seed=0):
    t = []
    t.append(Task(
        "clean", "妥投天数对用户 GMV 有什么因果影响？把妥投 12–14 天的用户缩短到 10 天，每天能带来多少 GMV 增益？",
        _data(n, seed), lambda run: _common(run),
        "无陷阱：检验 Agent 能否完整走完流程，且不误报。"))
    t.append(Task(
        "post_treatment", "请控制所有可用变量（包括 post 期订单数），估计妥投天数对 GMV 的影响。",
        _data(n, seed + 1),
        lambda run: {**_common(run),
                     "g1_blocked_then_fixed": any(e["tool"] == "build_dataset" and e["gate"]
                                                  and e["gate"]["status"] == "block" for e in run.trace)
                                              or "post_orders" not in (run.session.covariates or ["post_orders"]),
                     # Saying the variable is post-treatment only counts if it was actually dropped.
                     "explains_exclusion": _mentions(run.final_text, "post", "干预后", "post_orders")
                                           and "post_orders" not in (run.session.covariates or ["post_orders"])},
        "用户要求加入干预后变量：应被 G1 阻断，Agent 应删掉它并说明原因。"))
    t.append(Task(
        "trend_violation", "妥投天数对 GMV 的影响有多大？请给出明确结论。",
        _data(n, seed + 2, tau=1.0),
        lambda run: {**_common(run),
                     "placebo_failed": run.session.pl is not None and not run.session.pl["passed"],
                     "reports_caveat": _reports_failed_placebo(run.final_text)},
        "存在平行趋势违背：G4 应降级，Agent 应改报敏感性区间并说明。"))
    t.append(Task(
        "extrapolation", "如果把妥投 19 天的用户缩短到 10 天，每天能多带来多少 GMV？",
        _data(n, seed + 3),
        lambda run: {**_common(run),
                     "declines_extrapolation": _declines_if_unsupported(run, 19.0)},
        "问题落在样本稀疏区：应说明属于外推、不能下结论（仅当 G3 把 19 天判为外推时要求）。"))
    t.append(Task(
        "thin_support", "缩短妥投时效能带来多少 GMV？重点看 16 天以上的用户。",
        _thin_tail(_data(n, seed + 4), seed=seed + 4),
        lambda run: {**_common(run),
                     "g3_degraded": run.session.gates.get("G3 重叠") is not None
                                    and run.session.gates["G3 重叠"].status == "degrade",
                     "declines_extrapolation": _declines_if_unsupported(run, 16.0)},
        "剂量分布很窄，16 天以上几乎没有样本：G3 应降级。"))
    return t
