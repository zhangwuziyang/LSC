# E2 v2 方案（预注册）

回应三点评审意见：模拟不够丰富 → E1b；基准太小、评分规则改过多次 → E2 v2 + 预注册；需要证明 LLM 相对固定流程的价值 → 固定流程基线。

## 任务（14 个，`sim/traps_v2.py`）
| 类别 | 任务 | 考察 |
|---|---|---|
| 无陷阱 | clean_curve / clean_specific（11→10 天）/ clean_point_pressure（"不要区间"） | 不过度谨慎；回答所问的剂量；压力下仍说明不确定性 |
| 干预后变量 | post_explicit / post_labelled（中性名称，有时期标注）/ post_unlabelled（无时期标注，只有定义） | 语义层的作用；LLM 读定义的作用 |
| 处理泄漏 | leak_unlabelled（由处理推导的超时标记，无时期标注） | LLM 读定义的作用 |
| 平行趋势 | trend_strong（τ=1）/ trend_mild（τ=0.5）/ trend_pressure（"不用做检验"） | 检验与压力下的坚持 |
| 外推 | extrap_high（19→10）/ extrap_low（5→3）/ thin_support（16 天以上） | 两端外推 |
| 拐点 | kink_claim（"10 天是不是拐点"） | 不确定时不下结论 |

每次重复都重新生成数据。参考事实（外推剂量、拐点区间、placebo 结论）由参考引擎对同一份数据算一次（`experiments/e2v2_reference.py`），所有条件按同一事实评分。

## 条件
| 条件 | LLM | 引擎 | 检验 | 说明 |
|---|---|---|---|---|
| full | ✓ | ✓ | ✓（可见） | 完整系统 |
| no_gates | ✓ | ✓ | 计算但不给模型看 | 检验的作用 |
| code | ✓ | ✗ | ✗ | 大模型直接写代码 |
| fixed | ✗ | ✓ | ✓ | 固定流程：所有 pre 或未标注时期的变量、默认剂量对、模板报告 |

预期：fixed 在 post_unlabelled、leak_unlabelled（不会读定义）和 clean_specific、extrap_high/low（不回答所问剂量）上失败；在其他陷阱上与 full 相同。full 与 fixed 的差距就是 LLM 的额外价值；fixed 与 code 的差距就是引擎和检验的价值。

## 规模与成本（每家）
14 任务 × 3 次 × 3 个 LLM 条件 = 126 次运行（fixed 另有 42 次，不调用 API）。
- full / no_gates：每次约 2.5–5 万输入、0.3–1 万输出 token，约 5–7 分钟
- code：每次约 16–66 万输入、1–2.5 万输出 token，约 6–30 分钟
- 合计每家约 2,500 万输入 + 200 万输出 token（code 占九成）；两家并行约 10–14 小时

## 评分（预注册，`experiments/judging_v2/RUBRIC.md`）
两名独立盲评员 + 第三名仲裁；报告评分员 A、多数票两种口径和 Cohen's kappa。规则与任务自提交 `2c1c5d6` 起冻结。
