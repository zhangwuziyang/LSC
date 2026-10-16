# 项目完整方案：基于 LLM Agent 的连续处理因果推断系统（GPS-DID × AI）

> 版本 v0.1 ｜ 2026-10 ｜ 作者：zhangwuziyang
> 来源：实习汇报《妥投天数对用户 GMV 的影响》中的 GPS+DID 专题分析

---

## 0. 一页摘要

| 项目 | 内容 |
|---|---|
| 暂定题目 | **CausalAgent-GPS: An Assumption-Aware LLM Agent for Continuous-Treatment Difference-in-Differences, with Evidence from Cross-Border E-commerce Logistics** |
| 中文题目 | 面向连续处理双重差分的"假设感知型"大模型智能体：以跨境电商物流时效为例 |
| 核心问题 | 物流妥投天数（连续变量）如何因果地影响用户后续 GMV？如何让 AI 在**不牺牲统计严谨性**的前提下自动完成这类分析？ |
| 方法贡献 | 把原业务分析中的"正态 GPS + 差分结果"升级为 **ML 条件密度 GPS + 双稳健（DR）连续剂量响应 + DID**，并给出有效推断、重叠性诊断和平行趋势敏感性分析 |
| 系统贡献 | 一个三层 Agent 系统：语义层（指标库）→ 确定性估计引擎 → 假设把关（Gatekeeper）；LLM 只负责编排与解释，**不触碰数字** |
| 实证贡献 | 半合成基准（已知真值）+ 公开数据（Olist）+ 内部真实案例（脱敏、获批后） |
| 目标期刊/会议 | KDD Applied Data Science Track（首选）；CIKM Applied Research / WWW Industry；方法向可投 MSOM / POMS / Management Science |
| 周期 | 约 20 周（2026-10 → 2027-02） |

---

## 1. 研究背景与动机

### 1.1 业务背景
跨境电商中，物流履约时效（妥投天数）是影响用户体验与复购的关键因素。原分析（US 跨境线，140.7 万用户）发现：

- 剂量响应曲线随妥投天数增加整体下降，高价值档斜率更陡；
- **10 天**为拐点，超过后损失加速；
- 将 ≥12 天人群缩短至 10 天，日均 GMV 增益约 4.08 万美元（稳健版 2.91）。

### 1.2 为什么是连续处理
妥投天数是连续变量，传统 PSM 需人为二值化（损失剂量信息、每换一次阈值就要重新匹配）。GPS（Hirano & Imbens, 2004）把倾向得分推广到连续处理，可以回答"每多 1 天损失多少 GMV"。

### 1.3 为什么要结合 AI
- 因果分析流程长（取数 → 选混淆 → 建模 → 诊断 → 解读），高度依赖分析师经验，难以复用；
- 直接让 LLM 写代码做因果推断，会出现**选错变量（用到干预后变量）、忽略重叠性、跳过平行趋势检验、编造数字**等问题；
- 原工作已积累"AI 友好型指标库"（Markdown + YAML 语义层）和多个 Agent/Skill，天然适合作为系统底座。

---

## 2. 原方法回顾与不足（论文 Motivation 的直接来源）

### 2.1 原设计（三窗口队列）

| 窗口 | 时间 | 用途 |
|---|---|---|
| pre | 2026-02-14 ~ 04-30 | 用户特征（去混淆，GPS）与价值分层；pre 期 GMV |
| treatment | 2026-05-01 ~ 05-31 | 观测处理变量 D = 用户平均妥投天数 |
| post | 2026-06-01 ~ 08-15（76 天） | post 期 GMV；ΔY = Y_post − Y_pre（DID） |

### 2.2 六个问题

| # | 问题 | 证据 / 说明 | 本项目的解决方案 |
|---|---|---|---|
| P1 | GPS 正态假设不成立 | 全样本偏度 7.1、超额峰度 67.9；仅 6–15 天核心区近似正态 | ML 条件密度估计（§4.3） |
| P2 | 连续 DID 识别条件未讨论 | 不存在"未处理组"；跨剂量比较需要 *strong parallel trends*（Callaway, Goodman-Bacon & Sant'Anna, 2024） | 显式识别假设 + 敏感性界（§4.2, §4.7） |
| P3 | 重叠性分级阈值随意 | 2% / 0.5% 无理论依据 | 基于有效样本量（ESS）与 GPS 的正式重叠诊断与修剪（§4.6） |
| P4 | 增益矩阵无推断 | 曲线有 95% CI，但增益（两点差 × 人数）没有 | 双稳健估计 + 乘子 bootstrap，给出增益置信区间（§4.5） |
| P5 | 无平行趋势检验 | — | pre 期切分的 placebo 检验（§4.7） |
| P6 | 异质性靠手工分档 | 仅按 4 个价值档拆分 | 条件剂量响应 / 连续处理因果森林（§4.8） |

---

## 3. 研究问题

- **RQ1（方法）**：在处理变量重尾、偏态的情况下，ML-GPS + 双稳健 DID 相比正态 GPS 能在多大程度上降低剂量响应估计偏误、改善置信区间覆盖率？
- **RQ2（系统）**：一个"假设感知型" LLM Agent 能否自动完成连续处理因果分析，并且**在假设被违背时正确地拒绝或降级结论**？
- **RQ3（对比）**：与纯 LLM（直接写代码）和人工分析师相比，Agent 在估计准确性、假设违背检出率、数字幻觉率与耗时上的表现如何？
- **RQ4（实证/决策）**：物流时效对 GMV 的因果剂量响应形态如何？阈值（拐点）在哪里？在给定成本下应优先治理哪些人群？

---

## 4. 方法论

### 4.1 设定与记号
- 个体 i = 1..n；协变量 X_i（只取 pre 期及之前）；处理剂量 D_i ∈ 𝒟 ⊂ ℝ（treatment 期平均妥投天数）。
- 结果：Y_i,pre、Y_i,post；差分结果 ΔY_i = Y_i,post − Y_i,pre。
- 潜在结果：ΔY_i(d)。
- **估计目标**：
  - 平均剂量响应 μ(d) = E[ΔY(d)]；
  - 剂量对比 θ(d, d′) = μ(d′) − μ(d)，例如 θ(12, 10)；
  - 边际效应 μ′(d)（每延长 1 天的 GMV 变化）；
  - 分组版本 μ_g(d) = E[ΔY(d) | G = g]（g 为价值档）；
  - **业务增益**：对当前剂量为 d、治理到 d′ 的人群 S_d，
    日均增益 Gain(d→d′) = |S_d| · θ_{S_d}(d, d′) / 76。

### 4.2 识别假设
- **A1 条件强平行趋势（差分形式的弱无混淆）**：ΔY(d) ⫫ D | X，对所有 d ∈ 𝒟。
  含义：给定 X，不同剂量的用户如果接受同一剂量，GMV 变化趋势相同。这比"二元 DID 的平行趋势"更强，因为没有未处理组，必须比较不同剂量（CGBS 2024）。论文中需明确说明，并用 §4.7 做敏感性分析。
- **A2 重叠性**：条件密度 π(d | x) ≥ c > 0，对关注的 d 和几乎所有 x 成立。
- **A3 SUTVA / 无干扰**：一个用户的时效不影响其他用户的 GMV（跨用户的平台级溢出效应在讨论中说明）。
- **A4 无预期效应**：用户在 treatment 期之前不会因为"预期"未来的物流时效而改变 pre 期行为。
- **A5 混淆变量时序正确**：X 不能包含 treatment 期或之后的变量（Agent 的 Gatekeeper 会强制检查）。

关键混淆变量（候选）：pre 期历史平均妥投天数、收货地（州/邮编）、品类结构、商家/仓库类型、订单数、客单价、pre 期 GMV、活跃天数、注册时长、退款/投诉历史等。

### 4.3 GPS 估计：从正态到 ML 条件密度
| 方法 | 说明 | 角色 |
|---|---|---|
| M0 正态 GPS | D | X ~ N(Xβ, σ²)（Hirano & Imbens 2004；原业务做法） | 基线 |
| M1 异方差位置-尺度模型 | D = m(X) + s(X)·ε，m、s 用 GBM 拟合，ε 的密度用核估计 | 主力 |
| M2 混合密度网络 MDN | Bishop (1994)，K 个高斯分量 | 处理多峰 / 重尾 |
| M3 条件正态化流 | 可逆变换建模 π(d | x) | 复杂分布（可选） |
| M4 CBGPS | 协变量平衡 GPS（Fong, Hazlett & Imai 2018） | 以平衡为目标的替代方案 |
| 变换 | 对 D 取 log 或 Box-Cox 后再建模，最后换算回原尺度 | 降低偏度 |

模型选择标准：交叉验证的对数似然（条件密度的拟合度）+ 加权后的协变量平衡。

### 4.4 剂量响应估计量
1. **E0 Hirano–Imbens 两步法**（基线）：先用 ΔY 对 (D, GPS) 做多项式回归，再在每个 d 处对 GPS 取平均。
2. **E1 IPW 核估计**：用稳定化权重 w = f(D) / π(D | X) 做核加权回归。
3. **E2 双稳健估计（主估计量，Kennedy et al. 2017）**：
   - 拟合结果模型 m(x, d) = E[ΔY | X = x, D = d]（GBM / NN）与 GPS π(d | x)；
   - 构造伪结果
     ξ_i = (ΔY_i − m(X_i, D_i)) · ϖ(D_i) / π(D_i | X_i) + ∫ m(x, D_i) dP̂(x)，
     其中 ϖ(d) = ∫ π(d | x) dP̂(x)；
   - 用 ξ 对 D 做局部线性核回归，得到 μ̂(d)；
   - K 折交叉拟合（K = 5），带宽用留一交叉验证选择。
4. **E3 DML 连续处理**（Colangelo & Lee 2020）：作为稳健性对照。

性质：只要 m 和 π 中有一个估计得好，估计量就一致；两者都以 o(n^{-1/4}) 的速率收敛时，可以做有效推断。

### 4.5 推断
- 点态置信区间：Kennedy et al. (2017) 的方差估计；
- **一致置信带**与**增益矩阵 CI**：乘子 bootstrap（B = 1000），θ(d, d′) 与 Gain(d→d′) 用同一组 bootstrap 抽样，从而保留两点之间的相关性；
- 拐点推断：对 μ′(d) 做分段线性 / 门限回归拟合，用 bootstrap 给出拐点位置的 CI。

### 4.6 重叠性诊断（替代 2% / 0.5% 阈值）
- 每个剂量 d 的核加权**有效样本量** ESS(d) = (Σw)² / Σw²；
- GPS 修剪：去掉 π(d | X) < ε 的点，ε 由 Crump et al. (2009) 思路推广而来；
- 增益矩阵分级：以起点与终点剂量 ESS 的较小值为准，分为 *可信 / 谨慎 / 外推* 三级，并在图上标出。

### 4.7 平行趋势检验与敏感性分析
- **Placebo 检验**：把 pre 期切成 pre1（02-14 ~ 03-31）和 pre2（04-01 ~ 04-30），只用 pre1 的协变量，估计"未来剂量"D 对 Y_pre2 − Y_pre1 的影响，理论上应为 0；
- **敏感性分析**：参考 Rambachan & Roth (2023) 的思路，允许平行趋势存在幅度不超过 M · |placebo 偏离| 的违背，报告 θ 的识别区间，以及"结论被推翻所需的最小 M"（breakdown value）；
- 未观测混淆的敏感性：可选做 E-value 或 Rosenbaum 式的连续处理推广。

### 4.8 异质性
- 分组的双稳健伪结果回归：用 ξ 对 (D, G) 做回归，得到 μ̂_g(d)；
- 连续处理的因果森林（GRF local linear / 广义随机森林），得到 CATE 剂量曲线，找出"高敏感人群"，并与原工作中的"物流用户敏感度分群"对接。

### 4.9 决策层（可选扩展）
在成本 c(d→d′)（加线路、换承运商的成本）约束下：
max Σ Gain(d→d′) − c(d→d′)，求解在哪些人群、提速几天收益最大（整数规划，可接入 cuOpt / OR-Tools）。这一层把论文从"估计"延伸到"决策"。

---

## 5. 系统架构：CausalAgent-GPS

### 5.1 总体设计原则
1. **数字只来自确定性引擎**：LLM 不做任何计算；报告中的数字通过模板从引擎输出的 JSON 填入，并做校验和比对；
2. **假设前置把关**：每个阶段都有可执行的检验（Gate），未通过就阻断或降级，而不是"带病出结论"；
3. **语义层驱动**：变量含义、口径、时间粒度来自指标库 YAML，降低 LLM 误解口径的风险；
4. **可审计**：每次运行生成完整的 trace（调用了哪些工具、参数、结果、Gate 判定）。

### 5.2 架构图

```
                 ┌──────────────────────────────────────────┐
  用户自然语言 → │  Orchestrator（LLM，tool use）            │
  "妥投天数对     │  规划 → 调工具 → 读 Gate 结果 → 解释      │
   GMV 的影响？"  └───────┬───────────────┬──────────────────┘
                         │               │
       ┌─────────────────▼──┐   ┌────────▼──────────────────────┐
       │ ① 语义层            │   │ ② 确定性估计引擎（Python 包）  │
       │ 指标库 YAML         │   │ gpsdid.gps / estimators /     │
       │ - 变量定义/口径      │   │ inference / gains             │
       │ - 时间属性(pre/post) │   └────────┬──────────────────────┘
       │ - SQL 模板          │            │
       └─────────────────────┘   ┌────────▼──────────────────────┐
                                 │ ③ Gatekeeper（规则 + 统计检验）│
                                 │ G1 变量时序  G2 GPS 拟合       │
                                 │ G3 平衡性    G4 重叠性         │
                                 │ G5 Placebo   G6 敏感性         │
                                 └────────┬──────────────────────┘
                                          │
                                 ┌────────▼──────────────────────┐
                                 │ ④ 报告生成（模板 + LLM 解释）   │
                                 │ 数字校验 / 可信度分级 / 图表    │
                                 └───────────────────────────────┘
                                          │（可选）
                                 ┌────────▼──────────┐
                                 │ ⑤ 决策优化 (MILP)  │
                                 └───────────────────┘
```

### 5.3 工具清单（供 LLM 调用）

| 工具 | 输入 | 输出 | 关联 Gate |
|---|---|---|---|
| `metric_lookup` | 关键词 | 指标 YAML（定义、口径、表、时间属性） | — |
| `propose_confounders` | 处理变量、结果变量 | 候选混淆集合及理由 | G1 |
| `build_dataset` | 变量列表、窗口 | 分析表 + 数据画像 | G1 |
| `fit_gps` | 方法 M0–M4 | GPS、交叉验证似然 | G2 |
| `check_balance` | GPS | 加权前后各协变量与 D 的相关性 | G3 |
| `check_overlap` | GPS、剂量网格 | ESS(d)、修剪建议 | G4 |
| `placebo_test` | pre1/pre2 | placebo 曲线及检验 p 值 | G5 |
| `estimate_dose_response` | 估计量 E0–E3 | μ̂(d)、CI、一致置信带 | — |
| `sensitivity` | M 网格 | 识别区间、breakdown M | G6 |
| `gain_matrix` | 治理方案 | 增益矩阵 + CI + 可信度分级 | G4 |
| `heterogeneity` | 分组 / 森林 | μ̂_g(d) | — |
| `render_report` | 运行 trace | Markdown/HTML 报告 | 数字校验 |

### 5.4 Gate 规则示例

| Gate | 判定规则（默认值，可配置） | 未通过时的动作 |
|---|---|---|
| G1 时序 | 所有 X 的时间属性 ≤ pre 期结束 | **阻断**，要求替换变量 |
| G2 GPS 拟合 | 模型的交叉验证似然不劣于 M0；PIT 直方图检验 | 自动切换到更灵活的模型 |
| G3 平衡 | 加权后 \|corr(X_k, D)\| < 0.1 | 换模型 / 加交互项，仍失败则降级 |
| G4 重叠 | ESS(d) ≥ 500 为可信；100–500 为谨慎；< 100 为外推 | 标注或剔除该剂量区间 |
| G5 Placebo | 一致置信带覆盖 0 | 未通过则结论降级，并强制做 G6 |
| G6 敏感性 | breakdown M ≥ 1 | 报告识别区间，不报点估计结论 |

### 5.5 LLM 选型与部署
- 研究原型：Claude（Anthropic API，tool use）作为 Orchestrator；
- 对照：开源模型（如 Qwen 系列），用于内部部署场景和消融实验；
- 消融：去掉语义层 / 去掉 Gatekeeper / 让 LLM 自己写代码（不提供工具）。

---

## 6. 数据方案

| 数据 | 用途 | 说明 |
|---|---|---|
| **D1 半合成基准** | 主评测（有真值） | 用真实（或公开）协变量 X，人为设定 D 的生成过程（重尾、异方差）和 μ(d)（含 10 天拐点、异质性），控制混淆强度、重叠程度和平行趋势违背程度 |
| **D2 Olist 巴西电商公开数据** | 公开可复现的实证 | 含下单、发货、妥投时间、评价、商家与客户地理信息。**注意**：客户复购率很低（约 3%），建议用**商家-月面板**（剂量 = 商家当月平均妥投天数，结果 = 商家后续 GMV 变化）或"首单妥投天数 → 是否复购 / 评分"作为结果 |
| **D3 内部数据（US 跨境线）** | 真实案例 | 需获得公司书面审批；脱敏，只报告相对值或经缩放的数值；不将原始数据放入仓库 |

### 6.1 半合成数据生成过程（DGP）草案
```
X ~ 真实协变量（经 bootstrap 重采样）
D = exp( m(X) + s(X)·ε ),  ε ~ 0.8·N(0,1) + 0.2·t_3      # 重尾、右偏
μ(d) = −a·max(0, d−10)^1.5 − b·d                          # 10 天拐点
ΔY = μ(D)·h(G) + g(X) + τ·U + noise                       # h 为异质性；U 为可调的未观测混淆
场景网格：混淆强度 × 重叠程度 × 平行趋势违背 τ × 样本量 n ∈ {1e4, 1e5, 1e6}
```

### 6.2 注入式"陷阱"（用于评测 Agent）
1. 混淆候选中混入干预后变量（例如 post 期订单数）→ 应被 G1 阻断；
2. 制造重叠失败的区域（例如 18 天以上）→ 应被 G4 标注；
3. 注入平行趋势违背 → 应被 G5/G6 检出；
4. 结果变量单位或口径歧义 → 应通过语义层澄清；
5. 让 LLM 写报告时复述数字 → 检测数字幻觉率。

---

## 7. 实验设计

| 实验 | 目的 | 对比对象 | 指标 |
|---|---|---|---|
| **E1 估计量对比** | 回答 RQ1 | (M0, M1–M4) × (E0–E3) | μ(d) 的积分 RMSE、偏误、CI 覆盖率、CI 宽度、拐点误差 |
| **E2 重叠与修剪** | 验证 §4.6 | 固定阈值（2% / 0.5%）vs ESS 分级 | 增益矩阵误差、错误"高可用"占比 |
| **E3 Agent 端到端** | 回答 RQ2、RQ3 | 纯 LLM 写代码 / 不带 Gate 的 Agent / 完整 Agent / 人工分析师（3–5 人） | 估计误差、陷阱检出率、数字幻觉率、完成率、耗时、token 成本 |
| **E4 消融** | 拆解各组件的贡献 | −语义层、−Gatekeeper、−交叉拟合、不同 LLM | 同 E3 |
| **E5 实证** | 回答 RQ4 | 原业务结果 vs 新方法 | 剂量曲线、拐点 CI、增益矩阵 CI、异质性、placebo 与敏感性结果 |
| **E6 决策（可选）** | 估计 → 决策 | 按全口径增益排序 vs 带不确定性的优化 | 期望净收益、遗憾值（regret，在半合成数据上） |

每个场景做 200 次蒙特卡洛重复；Agent 实验每个任务做 5 次独立运行，报告均值与方差。

---

## 8. 代码仓库结构（本仓库）

```
LSC/
├── docs/
│   └── PROJECT_PLAN.md          # 本文档
├── gpsdid/                      # ② 确定性估计引擎（可独立发布为 pip 包）
│   ├── data.py                  # 分析表 schema、窗口切分
│   ├── gps/                     # normal.py, hetero_gbm.py, mdn.py, flow.py, cbgps.py
│   ├── estimators/              # hirano_imbens.py, ipw_kernel.py, dr_kennedy.py, dml.py
│   ├── inference.py             # 带宽选择、乘子 bootstrap、一致置信带
│   ├── diagnostics/             # balance.py, overlap.py, placebo.py, sensitivity.py
│   ├── heterogeneity.py
│   └── gains.py                 # 增益矩阵 + CI + 可信度分级
├── agent/                       # ①③④ Agent 系统
│   ├── semantic_layer/          # 指标库 YAML（公开版 / 脱敏版）
│   ├── tools.py                 # 工具定义（JSON schema）
│   ├── gatekeeper.py            # G1–G6
│   ├── orchestrator.py          # LLM tool-use 循环
│   ├── report.py                # 模板渲染 + 数字校验
│   └── prompts/
├── sim/
│   ├── dgp.py                   # 半合成 DGP
│   └── traps.py                 # 注入式陷阱
├── experiments/                 # E1–E6 脚本与配置（yaml）
├── data/                        # .gitignore；只放公开数据的下载脚本
├── paper/                       # LaTeX（ACM 模板）
└── tests/
```

技术栈：Python 3.11、numpy / pandas / scikit-learn / lightgbm / PyTorch（MDN、flow）/ statsmodels / econml（对照）/ anthropic SDK；实验用 hydra 或 yaml 管理配置。

---

## 9. 时间计划（约 20 周）

| 周 | 里程碑 | 产出 |
|---|---|---|
| W1–2 | 文献综述；定稿识别框架（§4.2） | 相关工作笔记、记号表 |
| W3–5 | 估计引擎 v1：M0/M1 + E0/E2 + bootstrap；单元测试 | `gpsdid` v0.1 |
| W6–7 | 半合成 DGP 与 E1/E2 实验 | 第一批图表（RMSE、覆盖率） |
| W8 | 诊断模块：重叠、placebo、敏感性 | `gpsdid` v0.2 |
| W9–11 | Agent 系统：语义层、工具、Gatekeeper、Orchestrator | `agent` v0.1，可端到端跑通 |
| W12–13 | E3/E4 Agent 评测（含人工分析师对照） | 评测表 |
| W14 | Olist 实证（E5） | 公开实证结果 |
| W15 | 内部数据复现（视审批情况） | 脱敏结果 |
| W16 | 决策层 E6（可选） | — |
| W17–19 | 论文写作、内部审阅 | 完整初稿 |
| W20 | 投稿；开源代码与基准 | 投稿 + GitHub release |

> KDD ADS 截稿时间通常在每年 2 月前后，请以官网公布为准；若赶不上，顺延至 CIKM（通常在 5 月）。

---

## 10. 论文结构（约 9 页 + 附录）

1. **Introduction**：业务问题 → 连续处理的挑战 → LLM 做因果推断的风险 → 三点贡献
2. **Related Work**：连续处理 GPS；连续处理 DID；双稳健 / DML；LLM 与因果推断；数据分析 Agent
3. **Problem Setup & Identification**：记号、估计目标、A1–A5、为什么需要强平行趋势
4. **Method**：ML-GPS、DR 剂量响应、推断、重叠诊断、敏感性
5. **System: CausalAgent-GPS**：架构、工具、Gatekeeper、数字校验
6. **Experiments**：E1–E4（半合成 + 陷阱）
7. **Case Study**：Olist + 跨境物流（E5，E6 可选）
8. **Discussion**：局限（SUTVA、平台级溢出、外部有效性）、部署经验
9. **Conclusion**
- 附录：DGP 细节、证明要点、提示词、完整结果表

---

## 11. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| 内部数据无法公开 | 实证贡献减弱 | 以半合成 + Olist 为主；内部数据只做脱敏案例或删去 |
| Olist 复购率低 | DID 样本不足 | 改用商家-月面板或其他结果变量；必要时换别的公开数据集 |
| 强平行趋势无法验证 | 审稿质疑识别 | placebo + 敏感性分析 + breakdown value，诚实报告 |
| 方法创新被认为是"已有方法组合" | 新颖性不足 | 突出系统贡献（Gatekeeper、陷阱基准）与实证结论；基准开源 |
| LLM 结果不稳定 | 可复现性差 | 固定温度、多次运行报告方差；数字只来自引擎 |
| 大样本（10⁶）下 bootstrap 太慢 | 工程瓶颈 | 乘子 bootstrap + 交叉拟合结果复用；可用 GPU |

---

## 12. 立即可开始的下一步

1. 建 `gpsdid` 包骨架，实现 M0（正态 GPS）+ E0（Hirano–Imbens），先复现原业务流程；
2. 写 `sim/dgp.py`，用半合成数据验证"正态 GPS 在重尾下有偏"（RQ1 的第一张图）；
3. 下载 Olist，评估商家-月面板的样本量与剂量分布；
4. 整理一份公开版的指标库 YAML（基于 Olist 字段）作为语义层样例。

---

## 参考文献（核心）

- Hirano, K., & Imbens, G. W. (2004). The propensity score with continuous treatments. In *Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives*. Wiley.
- Imai, K., & van Dyk, D. A. (2004). Causal inference with general treatment regimes. *JASA*, 99(467).
- Kennedy, E. H., Ma, Z., McHugh, M. D., & Small, D. S. (2017). Non-parametric methods for doubly robust estimation of continuous treatment effects. *JRSS-B*, 79(4).
- Colangelo, K., & Lee, Y.-Y. (2020). Double debiased machine learning nonparametric inference with continuous treatments. arXiv:2004.03036.
- Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2024). Difference-in-differences with a continuous treatment. NBER Working Paper 32117 / arXiv:2107.02637.
- Abadie, A. (2005). Semiparametric difference-in-differences estimators. *Review of Economic Studies*, 72(1).
- Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences estimators. *Journal of Econometrics*, 219(1).
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1).
- Fong, C., Hazlett, C., & Imai, K. (2018). Covariate balancing propensity score for a continuous treatment. *Annals of Applied Statistics*, 12(1).
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited overlap in estimation of average treatment effects. *Biometrika*, 96(1).
- Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends. *Review of Economic Studies*, 90(5).
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *JASA*, 113(523).
- Bishop, C. M. (1994). Mixture density networks. Technical report, Aston University.
- Kıcıman, E., Ness, R., Sharma, A., & Tan, C. (2023). Causal reasoning and large language models: Opening a new frontier for causality. arXiv:2305.00050.
- Olist. Brazilian E-Commerce Public Dataset by Olist. Kaggle.
