# 系统流程图

蓝色 = LLM 环节，橙色 = 假设把关 Gate，绿色 = 输出。

```mermaid
flowchart TD
    Q["用户提问<br/>妥投天数如何影响 GMV？"] --> O["LLM Orchestrator<br/>规划 · 调用工具 · 解释"]

    O --> S["① 语义层<br/>指标库 YAML：口径 / 表 / 时间属性"]
    S --> B["构建分析表<br/>X: pre 期特征 · D: 5月妥投天数 · ΔY = post − pre GMV"]
    B --> G1{"G1 变量时序<br/>X 均早于干预期？"}
    G1 -- 否 --> B

    G1 -- 是 --> GPS["② 估计 GPS<br/>正态基线 vs ML 条件密度"]
    GPS --> G23{"G2 拟合 / G3 平衡<br/>加权后 |corr(X,D)| < 0.1？"}
    G23 -- 否 --> GPS

    G23 -- 是 --> G4["G4 重叠诊断<br/>ESS(d) 分级：可信 / 谨慎 / 外推"]
    G4 --> DR["双稳健剂量响应 μ(d)<br/>交叉拟合 + bootstrap 置信带"]
    DR --> G5{"G5 Placebo<br/>pre1→pre2 无效应？"}
    G5 -- 否 --> G6["G6 敏感性分析<br/>识别区间 · breakdown M"]
    G5 -- 是 --> OUT
    G6 --> OUT["输出<br/>剂量曲线 · 拐点 · 增益矩阵 + CI · 异质性"]

    OUT --> R["④ 报告生成<br/>数字仅来自引擎 · 可信度分级"]
    R --> DEC["⑤ 决策优化（可选）<br/>成本约束下选择治理人群"]

    classDef gate fill:#fff4e5,stroke:#e69500,color:#000;
    classDef llm fill:#e8f0fe,stroke:#3b6fd8,color:#000;
    classDef out fill:#e6f4ea,stroke:#2e8b57,color:#000;
    class G1,G23,G4,G5,G6 gate;
    class O,R llm;
    class OUT,DEC out;
```
