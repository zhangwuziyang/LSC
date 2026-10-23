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
| `tests/` | 单元测试 |

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
