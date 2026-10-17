"""Gates G1-G5. Each returns a GateResult; the pipeline decides what a status means.

status: "pass" | "degrade" (continue, but weaken the conclusion) | "block" (stop)
"""
import re
from dataclasses import asdict, dataclass

import numpy as np

from gpsdid import balance_ok


@dataclass
class GateResult:
    gate: str
    status: str
    detail: str

    def to_dict(self):
        return asdict(self)


def g1_timing(covariates, semantic):
    bad = [c for c in covariates if semantic.period(c) != "pre"]
    if bad:
        info = ", ".join(f"{c} (period={semantic.period(c)})" for c in bad)
        return GateResult("G1 时序", "block", f"以下变量不在 pre 期，不能作为混淆变量：{info}"), bad
    return GateResult("G1 时序", "pass", f"{len(covariates)} 个协变量均为 pre 期变量"), []


def g2_balance(table, threshold=0.1):
    worst = table.loc[table.corr_weighted.abs().idxmax()]
    msg = (f"加权后最大 |corr(X, D)| = {abs(worst.corr_weighted):.3f}（{worst.covariate}），"
           f"加权前为 {abs(table.corr_raw).max():.3f}，阈值 {threshold}")
    return GateResult("G2 平衡", "pass" if balance_ok(table, threshold) else "degrade", msg)


def g3_overlap(grid, grades):
    ext = grid[grades == "extrapolation"]
    if len(ext) == 0:
        return GateResult("G3 重叠", "pass", "所有剂量点的有效样本量都足够")
    return GateResult("G3 重叠", "degrade",
                      f"{len(ext)} 个剂量点（{ext.min():g}–{ext.max():g} 天范围内）有效样本量不足，"
                      "这些点只作外推参考，不进入结论")


def g4_placebo(passed, max_dev):
    if passed:
        return GateResult("G4 Placebo", "pass", f"placebo 曲线的一致置信带覆盖 0（最大偏离 {max_dev:.3f}）")
    return GateResult("G4 Placebo", "degrade",
                      f"placebo 曲线显著偏离 0（最大偏离 {max_dev:.3f}），平行趋势可能不成立，只报告敏感性区间")


_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def g5_numbers(text, allowed, rel_tol=0.005):
    """Every number in `text` must match a number the engine produced (or a whitelisted
    constant such as dates and thresholds), up to display rounding."""
    allowed = np.asarray(sorted(set(float(a) for a in allowed)))
    unmatched = []
    for tok in _NUM.findall(text):
        v = float(tok)
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        tol = max(0.5 * 10 ** -decimals, rel_tol * abs(v))
        if not np.any(np.abs(allowed - v) <= tol):
            unmatched.append(tok)
    if unmatched:
        return GateResult("G5 数字核对", "block", f"报告中有 {len(unmatched)} 个数字在引擎输出中找不到：{unmatched[:10]}")
    return GateResult("G5 数字核对", "pass", "报告中的每个数字都能在引擎输出中找到")
