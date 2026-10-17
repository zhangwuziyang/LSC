"""Template report from `run_analysis` output, checked by Gate G5 before release."""
import numpy as np

from .gates import g5_numbers

GRADE_CN = {"trusted": "可信", "caution": "谨慎", "extrapolation": "外推"}
STATUS_CN = {"pass": "通过", "degrade": "降级", "block": "阻断"}


def _numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _numbers(v)
    elif isinstance(obj, (int, float, np.integer, np.floating)) and not isinstance(obj, bool):
        yield float(obj)


def allowed_numbers(result):
    nums = list(_numbers({k: v for k, v in result.items() if k != "gates"}))
    # Display constants used by the template itself.
    nums += [95, 0.1, 1, 2, 3, 4, 5]
    return nums


def render(result, narrative=""):
    """Markdown report. `narrative` is optional LLM-written text; it is checked too."""
    L = ["# 物流时效对 GMV 的因果影响分析", ""]
    L += ["## 假设检验", "", "| Gate | 结果 | 说明 |", "|---|---|---|"]
    L += [f"| {g.gate} | {STATUS_CN[g.status]} | {g.detail} |" for g in result["gates"]]
    L.append("")
    if any(g.status == "block" for g in result["gates"]):
        L.append("**分析被阻断，没有输出结论。** 请按上表修正后重跑。")
        return "\n".join(L), None

    c, d0, days = result["curve"], result["d0"], result["days"]
    degraded = not result["placebo_passed"]
    L += ["## 剂量响应", "",
          f"相对 {d0:g} 天的日均 GMV 变化 θ({d0:g}, d)，95% 置信区间，样本量 {result['n']}。", "",
          "| 妥投天数 | θ | 95% CI | 可信度 |" + (" 敏感性区间 (M=1) |" if degraded else ""),
          "|---|---|---|---|" + ("---|" if degraded else "")]
    for i, d in enumerate(c["d"]):
        if d % 1 or d == d0:
            continue
        row = f"| {d:g} | {c['theta'][i]:.3f} | [{c['lo'][i]:.3f}, {c['hi'][i]:.3f}] | {GRADE_CN[c['grade'][i]]} |"
        if degraded:
            row += f" [{c['sens_lo'][i]:.3f}, {c['sens_hi'][i]:.3f}] |"
        L.append(row)
    k = result["kink"]
    L += ["", f"拐点估计为 {k['point']:.2f} 天，95% CI [{k['lo']:.2f}, {k['hi']:.2f}]。", ""]

    L += ["## 增益矩阵", "", f"把当前妥投 a 天的用户缩短到 b 天，带来的日均 GMV 增益（post 期按 {days} 天折算）。"
          "外推格子不进入结论。", "",
          "| a → b | 人数 | 日均增益 | 95% CI | 可信度 |", "|---|---|---|---|---|"]
    for g in result["gains"]:
        if g["grade"] == "extrapolation":
            continue
        L.append(f"| {g['from_day']:g} → {g['to_day']:g} | {g['n_units']} | {g['gain_per_day']:.2f} | "
                 f"[{g['gain_lo']:.2f}, {g['gain_hi']:.2f}] | {GRADE_CN[g['grade']]} |")
    if degraded:
        L += ["", "**注意：placebo 检验未通过，上述点估计可能有偏，结论以敏感性区间为准。**"]
    if narrative:
        L += ["", "## 解读", "", narrative]
    text = "\n".join(L)
    # Gate details are engine-generated strings; only the body after them is checked.
    body = text.split("## 剂量响应", 1)[1]
    return text, g5_numbers(body, allowed_numbers(result) + [d0, days])
