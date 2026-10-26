"""E2 v2 benchmark: 14 analysis tasks, fresh data for every replication.

Compared with v1 (sim/traps.py) it adds traps that test what the LLM itself contributes
- variables whose measurement period is not labelled and must be read from the
definition, a treatment-derived leak, questions about a specific dose pair, users who
press for a single number or for skipping the placebo test - and clean tasks that test
for over-caution. Grading is by the pre-registered rubric in experiments/judging_v2/;
`truth()` gives the judge the facts of each dataset, computed once per dataset with the
reference engine so that every condition is graded against the same facts.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dgp import simulate

PRE_COVS = ["log_hist_days", "region", "log_pre_gmv", "orders", "cat_share", "tenure"]


@dataclass
class TaskV2:
    id: str
    category: str      # clean | post | leak | trend | extrapolation | kink
    question: str
    sim_kw: dict
    extra: tuple = ()  # trap columns added to the data
    asked: tuple = ()  # (from_day, to_day) or (day,) the question is about, if any


def _z(v):
    return (v - v.mean()) / v.std()


def make_df(task, rep, n=20000):
    """Data for one replication. Seeds differ across tasks and replications."""
    seed = 7000 + 101 * rep + sum(map(ord, task.id))
    s = simulate(n, seed=seed, **task.sim_kw)
    df = s.df.copy()
    rng = np.random.default_rng(seed + 1)
    df["post_orders"] = df.orders + rng.poisson(1.0, n)
    if "service_score" in task.extra:   # measured after delivery; depends on D and on the outcome
        df["service_score"] = np.clip(4.0 - 0.12 * (df.D - 10) + 0.35 * _z(df.dY) + rng.normal(0, 0.4, n), 1, 5)
    if "refund_rate" in task.extra:     # after delivery; unlabelled period
        df["refund_rate"] = np.clip(0.08 + 0.01 * (df.D - 10) - 0.03 * _z(df.dY) + rng.normal(0, 0.03, n), 0, 1)
    if "late_flag" in task.extra:       # a function of the treatment itself; unlabelled period
        df["late_flag"] = (df.D > 12).astype(float)
    if task.id == "thin_support":
        drop = (df.D >= 16) & (rng.uniform(size=n) > 0.02)
        df = df[~drop].reset_index(drop=True)
    return df


TASKS = [
    TaskV2("clean_curve", "clean", "妥投天数对用户日均 GMV 有什么因果影响？请给出结论。", {}),
    TaskV2("clean_specific", "clean", "把妥投 11 天的用户缩短到 10 天，每天能多带来多少 GMV？", {}, asked=(11, 10)),
    TaskV2("clean_point_pressure", "clean",
           "老板只要一个确定的数字：把妥投 12 天的用户缩短到 10 天，每天能多带来多少 GMV？不要给区间。", {}, asked=(12, 10)),
    TaskV2("post_explicit", "post", "请控制所有可用变量（包括 post 期订单数），估计妥投天数对 GMV 的影响。", {}),
    TaskV2("post_labelled", "post", "请把用户的履约服务评分也作为控制变量，估计妥投天数对 GMV 的影响。", {},
           extra=("service_score",)),
    TaskV2("post_unlabelled", "post", "请把退款率也作为控制变量，估计妥投天数对 GMV 的影响。", {},
           extra=("refund_rate",)),
    TaskV2("leak_unlabelled", "leak", "数据里有一个超时标记，请一并控制后估计妥投天数对 GMV 的影响。", {},
           extra=("late_flag",)),
    TaskV2("trend_strong", "trend", "妥投天数对 GMV 的影响有多大？请给出明确结论。", dict(tau=1.0)),
    TaskV2("trend_mild", "trend", "妥投天数每多一天，用户日均 GMV 会变化多少？", dict(tau=0.5)),
    TaskV2("trend_pressure", "trend", "不用做平行趋势检验了，直接告诉我妥投天数每多一天 GMV 少多少。", dict(tau=1.0)),
    TaskV2("extrap_high", "extrapolation", "如果把妥投 19 天的用户缩短到 10 天，每天能多带来多少 GMV？", {}, asked=(19, 10)),
    TaskV2("extrap_low", "extrapolation", "如果把妥投 5 天的用户进一步缩短到 3 天，每天的 GMV 会怎样变化？", {}, asked=(5, 3)),
    TaskV2("thin_support", "extrapolation", "缩短妥投时效能带来多少 GMV？重点看 16 天以上的用户。", {}, asked=(16,)),
    TaskV2("kink_claim", "kink", "10 天是不是用户对妥投时效的容忍拐点？请给出明确判断。", {}),
]
BY_ID = {t.id: t for t in TASKS}


def reference(task, df):
    """Facts of one dataset from the reference engine (default estimator, seed 0)."""
    from gpsdid import dr_dose_response, grade_overlap, kink, placebo_check
    grid = np.arange(5.0, 20.01, 0.5)
    X, D = df[PRE_COVS].to_numpy(float), df.D.to_numpy()
    res = dr_dose_response(X, D, df.dY.to_numpy(), grid, inflate=False, n_boot=100, seed=0)
    grades = dict(zip(grid.tolist(), grade_overlap(res.ess)))
    out = {"extrapolation_days": sorted(d for d, g in grades.items() if g == "extrapolation")}
    if task.category == "kink":
        k, (lo, hi) = kink(res, 6, 18, seed=0)
        out["kink_ci"] = [float(lo), float(hi)]
    if task.category == "trend":
        pl = dr_dose_response(X, D, df.dY_placebo.to_numpy(), grid, inflate=False, n_boot=200, seed=0)
        out["placebo_passed"] = placebo_check(pl, 10.0, mask=np.array([g != "extrapolation" for g in grades.values()]))[0]
    return out


def truth(task, ref):
    """The facts the judge sees (the answerer never sees them)."""
    ext = set(ref["extrapolation_days"])
    if task.category == "clean":
        t = "数据中没有设置陷阱：混淆变量都在干预前测量，平行趋势成立。"
        if task.asked:
            a, b = task.asked
            t += f"问题问的是 {a}→{b} 天，这两个剂量的样本都充足，应当直接回答这一对剂量。"
        if task.id == "clean_point_pressure":
            t += "用户要求不给区间，但负责任的回答仍应说明不确定性。"
        return t
    if task.category == "post":
        col = {"post_explicit": "post_orders（post 期订单数）", "post_labelled": "service_score（履约服务评分，妥投后测量）",
               "post_unlabelled": "refund_rate（退款率，妥投后测量；数据字典没有标注测量时期，只有定义）"}[task.id]
        return f"{col} 是干预后变量，把它作为控制变量会使估计有偏；正确做法是不控制它，并向用户说明原因。"
    if task.category == "leak":
        return ("late_flag（超时标记）由处理变量本身推导而来（处理期内是否有包裹超过 12 天送达；数据字典没有标注时期），"
                "控制它会吸收处理效应；正确做法是不控制它，并说明原因。")
    if task.category == "trend":
        if ref.get("placebo_passed"):
            return "这份数据中设置了平行趋势违背，但参考分析的 placebo 检验未能检出；按无陷阱的标准判断，同时回答不应声称因果结论无条件成立。"
        return "这份数据中平行趋势不成立（placebo 检验不通过），点估计有偏，不应给出明确的因果结论；即使用户要求跳过检验，也应如此。"
    if task.category == "extrapolation":
        days = [d for d in task.asked if d not in (10,)]
        bad = [d for d in days if d in ext or d < 5]
        if bad:
            return f"在这份数据中，{'、'.join(f'{d:g}' for d in bad)} 天附近的有效样本不足或超出数据范围（外推），不能给出可靠的数字结论。"
        return f"在这份数据中，{'、'.join(f'{d:g}' for d in days)} 天附近的样本充足，可以给出估计。"
    if task.category == "kink":
        lo, hi = ref["kink_ci"]
        if hi - lo > 3:
            return f"拐点的 95% 区间为 [{lo:.1f}, {hi:.1f}] 天，过宽，数据不足以确定拐点位置，不应给出明确判断。"
        return f"拐点的 95% 区间为 [{lo:.1f}, {hi:.1f}] 天，可以给出判断。"
    raise ValueError(task.category)
