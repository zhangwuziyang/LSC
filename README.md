# GPS-DID × AI

研究计划见 [docs/proposal.html](docs/proposal.html)，流程图见 [docs/FLOWCHART.md](docs/FLOWCHART.md)，方法修正与实验结果见 [docs/RESULTS.md](docs/RESULTS.md)。

## 代码结构

| 路径 | 内容 |
|---|---|
| `gpsdid/gps.py` | GPS 模型：`NormalGPS`（M0，原业务做法）、`HeteroGBMGPS`（M1，log 变换 + 异方差 LightGBM + 残差核密度） |
| `gpsdid/estimators.py` | 剂量响应估计：`hirano_imbens`（E0）、`dr_dose_response`（E1，双稳健 + 5 折交叉拟合） |
| `gpsdid/inference.py` | 局部线性回归、留一交叉验证选带宽、乘子 bootstrap、一致置信带 |
| `gpsdid/diagnostics.py` | Gate 用的诊断：加权平衡表、ESS 重叠分级、placebo 检验、敏感性区间 |
| `gpsdid/gains.py` | 拐点估计（带 CI）、增益矩阵（带 CI 和可信度分级） |
| `sim/dgp.py` | 半合成数据：右偏重尾的妥投天数、10 天拐点、价值档异质性、可调的趋势违背 |
| `experiments/e1_estimators.py` | 实验 E1：M0/M1 × E0/E1 四种组合的蒙特卡洛对比 |
| `agent/session.py` | 分析会话：6 个步骤（建表、GPS、剂量响应、placebo、报告），每步带 Gate；也是 Agent 的工具 |
| `agent/orchestrator.py` | LLM 编排器：Claude 通过工具调用完成分析，所有数字由引擎产生并经 G5 核对 |
| `agent/pipeline.py` | 固定顺序的确定性流程（与 Agent 共用同一套步骤） |
| `sim/traps.py` | E2 埋错基准：5 个任务及自动评分 |
| `experiments/e2a_placebo_gate.py` | E2a：placebo 检验 G4 的误报率与检出率 |
| `experiments/e2b_agent.py` | E2b：Agent 在埋错基准上的表现（完整系统 vs 关闭 Gate / 关闭语义层） |
| `tests/` | 单元测试（含用假客户端测试 Agent 循环） |

## 快速开始

```bash
pip install -e ".[dev]"
pytest -q
python experiments/e1_estimators.py --reps 30 --n 20000   # 结果写到 experiments/results/
```

```python
import numpy as np
from gpsdid import dr_dose_response, balance_table, kink, gain_matrix
from sim.dgp import simulate

s = simulate(20000)
grid = np.arange(5, 20.01, 0.5)
res = dr_dose_response(s.X, s.D, s.dY, grid)        # 剂量响应 + bootstrap
theta, draws = res.contrast(10)                      # θ(10, d)
print(balance_table(s.X, s.D, res.weights))          # G2：加权平衡
print(kink(res, 6, 18))                              # 拐点及 95% CI
print(gain_matrix(res, s.D, from_days=[12, 13, 14], to_days=[10, 11]))
```

## 运行 Agent（需要 Anthropic API 凭证）

在环境变量中设置 `ANTHROPIC_API_KEY`（云端环境在环境设置里添加，新会话生效），然后：

```bash
python experiments/e2b_agent.py --conditions full --tasks clean --reps 1   # 先跑一个任务确认配置
python experiments/e2b_agent.py --reps 3                                   # 完整 E2b：3 种条件 × 5 个任务 × 3 次
```

默认模型为 `claude-opus-5-5`，开启了服务端拒答回退（`fallbacks="default"`）。

也可以用通义千问或 DeepSeek：

| provider | 接口 | key 变量 | 默认模型 | 需要放行的域名 |
|---|---|---|---|---|
| `qwen` | 千问 AI 平台，Anthropic 兼容 | `QWEN_API_KEY`（或 `DASHSCOPE_API_KEY`） | `qwen3.8-max`（`QWEN_MODEL` 可改） | `maas.qianwenaiapi.com` |
| `deepseek` | OpenAI 兼容 | `DEEPSEEK_API_KEY` | `deepseek-v4-pro`（`DEEPSEEK_MODEL` 可改） | `api.deepseek.com` |

key 放在云端环境设置里，或复制 `.env.example` 为 `.env` 填写（`.env` 不会提交）。先检查连通性和模型名，再跑实验：

```bash
python -m agent.check_providers
python experiments/e2b_agent.py --provider qwen --reps 3
python experiments/e2b_agent.py --provider deepseek --reps 3
python experiments/e2c_compare.py        # 两家对比
```
