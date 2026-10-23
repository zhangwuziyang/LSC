"""The orchestrator loop against a scripted fake client (no API calls)."""
import json
from types import SimpleNamespace as NS

import numpy as np
from lightgbm import LGBMRegressor

from agent.orchestrator import TOOLS, run_agent
from agent.pipeline import run_analysis
from gpsdid import HeteroGBMGPS
from sim.traps import PRE_COVS, make_tasks


def fast_gps():
    return HeteroGBMGPS(mean_kw=dict(n_estimators=60, n_jobs=1), scale_kw=dict(n_estimators=40, n_jobs=1))


def fast_outcome(seed=0):
    return LGBMRegressor(n_estimators=80, num_leaves=15, n_jobs=1, random_state=seed, verbose=-1)


FAST = dict(estimator_kw=dict(gps=fast_gps, outcome=fast_outcome, n_folds=3, n_boot=100, inflate=False))


class FakeClient:
    """Plays back a fixed list of assistant turns and records the tool results it gets."""

    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        assert kw["model"] == "claude-opus-5-5" and kw["fallbacks"] == "default"
        last = kw["messages"][-1]
        if isinstance(last["content"], list):
            self.seen.append([json.loads(r["content"]) for r in last["content"]])
        step = self.script.pop(0)
        if isinstance(step, str):
            return NS(stop_reason="end_turn", content=[NS(type="text", text=step)], usage=None)
        blocks = [NS(type="tool_use", id=f"t{len(self.seen)}_{i}", name=n, input=a) for i, (n, a) in enumerate(step)]
        return NS(stop_reason="tool_use", content=blocks, usage=None)


def test_tools_are_strict_schemas():
    for t in TOOLS:
        assert t["strict"] and t["input_schema"]["additionalProperties"] is False


def test_agent_loop_handles_block_and_number_check():
    task = {t.id: t for t in make_tasks(3000)}["post_treatment"]
    script = [
        [("list_metrics", {})],
        [("build_dataset", {"covariates": PRE_COVS + ["post_orders"]})],
        [("build_dataset", {"covariates": PRE_COVS})],
        [("fit_gps", {})],
        [("estimate_dose_response", {"from_days": [12, 13], "to_days": [10]})],
        [("run_placebo", {})],
        [("render_report", {"narrative": "缩短到 10 天每天可增加 123.456 美元。"})],
        [("render_report", {"narrative": "post_orders 是干预后变量，已从混淆变量中删除。"})],
        "已删除干预后变量 post_orders，报告见上。",
    ]
    client = FakeClient(script)
    run = run_agent(task.df, task.question, client=client, session_kw=FAST)
    assert run.stop_reason == "end_turn" and not client.script
    assert client.seen[1][0]["gate"]["status"] == "block"            # G1 blocked post_orders
    assert client.seen[6][0]["gate"]["status"] == "block"            # G5 caught 123.456
    assert client.seen[7][0]["gate"]["status"] == "pass"
    checks = task.checks(run)
    assert all(checks.values()), checks


def test_step_order_errors_are_tool_errors():
    task = make_tasks(2000)[0]
    client = FakeClient([[("fit_gps", {})], "stop"])
    run = run_agent(task.df, "q", client=client, session_kw=FAST)
    assert run.trace[0]["is_error"] and "build_dataset" in client.seen[0][0]["error"]


def test_pipeline_matches_session_path():
    task = make_tasks(3000)[0]
    out = run_analysis(task.df, covariates=PRE_COVS, **FAST)
    assert [g.gate for g in out["gates"]] == ["G1 时序", "G2 平衡", "G3 重叠", "G4 Placebo"]
    assert np.isfinite(out["curve"]["theta"]).all()
