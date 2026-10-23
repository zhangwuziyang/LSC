# 会话交接（2026-10-23）

新会话从这里接着做。分支：`claude/gps-did-causal-inference-x1sa9f`。

## 当前状态
- 方法部分已完成并通过验证，结果见 `docs/RESULTS.md`（E1 v2：新方法 RMSE 0.064，95% CI 覆盖率 96.5%）。
- Agent 系统已完成：`agent/session.py`（6 个工具步骤 + G1–G5）、`agent/orchestrator.py`（Claude / DeepSeek / 通义千问）、`sim/traps.py`（5 个埋错任务）。22 个测试通过。
- 千问用的是「千问 AI 平台」（platform.qianwenai.com），不是百炼；接口改为 Anthropic 兼容地址 `https://maas.qianwenaiapi.com/apps/anthropic`（代码默认值），需要放行 `maas.qianwenaiapi.com`。key 读 `QWEN_API_KEY`，没有则读 `DASHSCOPE_API_KEY`。
- 第一次连通检查（session_01SNYrEzQLxntqNkhSA4wsZU）：两家都 401。DeepSeek key 长度 117，疑似粘贴错误；千问 key 发到了百炼地址。
- 用户指定的旗舰模型：`qwen3.8-max`、`deepseek-v4.1-pro`（模型 ID 写法待 check_providers 确认）。

## 新会话要做的事（按顺序）
1. 安装依赖：`pip install -e ".[dev,agent]"`
2. 检查两家的 key、模型名和工具调用：`python -m agent.check_providers`。
   如果模型名不对，按输出里的候选名设置 `QWEN_MODEL` / `DEEPSEEK_MODEL`。
3. 各跑一个任务确认流程：
   `python experiments/e2b_agent.py --provider qwen --tasks clean --reps 1`
   `python experiments/e2b_agent.py --provider deepseek --tasks clean --reps 1`
   DeepSeek 推理模型的 `reasoning_content` 回传方式未实测，若报 400 需在 `_loop_openai_compat` 里调整。
4. 和用户确认费用后跑完整 E2b（每家 3 条件 × 5 任务 × 3 次 = 45 次运行）：
   `python experiments/e2b_agent.py --provider qwen --reps 3`
   `python experiments/e2b_agent.py --provider deepseek --reps 3`
   每次运行包含两次带方差膨胀的估计（约 3–4 分钟），单次后台命令上限 2 小时，按 provider 或 condition 拆开跑。
5. `python experiments/e2c_compare.py` 出对比表，告诉用户哪家效果更好。

## 未完成的实验

> **更新：E2a 由旧会话继续跑完并推送，新会话不要再跑 E2a**，以免两个会话同时提交同一个结果文件。推送前先 `git pull --rebase`。

- E2a（placebo 检验 G4 的误报率/检出率）：旧会话跑了一部分，结果在 `experiments/results/e2a_placebo_taus0_0.25.csv`。
  需要补跑：`python experiments/e2a_placebo_gate.py --taus 0 0.25 --reps 15`（如已有部分结果可只跑缺的）以及
  `python experiments/e2a_placebo_gate.py --taus 0.5 1 --reps 15`。每次重复约 4 分钟，注意 2 小时上限。

## 注意
- 不要同时跑两个重计算任务：LightGBM 多线程会互相抢 CPU，速度慢 5 倍以上。
- 不要把 API key 写进仓库；`.env` 已被 git 忽略。
