import numpy as np
import pandas as pd

from agent.gates import g1_timing, g2_balance, g5_numbers
from agent.semantic import SemanticLayer


def test_semantic_lookup_and_periods():
    s = SemanticLayer()
    assert s.lookup("妥投时效")[0]["column"] == "D"
    assert s.period("post_orders") == "post" and s.period("orders") == "pre"


def test_g1_blocks_post_treatment_covariate():
    s = SemanticLayer()
    ok, _ = g1_timing(["orders", "tenure"], s)
    bad, cols = g1_timing(["orders", "post_orders"], s)
    assert ok.status == "pass" and bad.status == "block" and cols == ["post_orders"]


def test_g2_degrades_on_imbalance():
    tbl = pd.DataFrame(dict(covariate=["a", "b"], corr_raw=[0.6, 0.2], corr_weighted=[0.25, 0.01]))
    assert g2_balance(tbl).status == "degrade"
    assert g2_balance(tbl.assign(corr_weighted=[0.05, 0.01])).status == "pass"


def test_g5_number_check():
    allowed = [0.31234, -1.5, 4700]
    assert g5_numbers("θ = 0.312，CI [-1.50, 0.40]，人数 4700", allowed + [0.4]).status == "pass"
    res = g5_numbers("增益为 3.41 万美元", allowed)
    assert res.status == "block" and "3.41" in res.detail


def test_g5_reads_unicode_minus():
    allowed = [-2.5111, 2.2096, -0.2902]
    text = "CI −2.5111 至 2.2096，增益 －0.2902"
    assert g5_numbers(text, allowed).status == "pass"
    assert g5_numbers("CI −9.99", allowed).status == "block"


def test_g5_accepts_numbers_from_any_tool_output():
    from sim.traps import PRE_COVS, make_tasks
    from agent.session import AnalysisSession
    from lightgbm import LGBMRegressor
    from gpsdid import HeteroGBMGPS
    kw = dict(gps=lambda: HeteroGBMGPS(mean_kw=dict(n_estimators=40, n_jobs=1), scale_kw=dict(n_estimators=30, n_jobs=1)),
              outcome=lambda seed=0: LGBMRegressor(n_estimators=60, n_jobs=1, verbose=-1),
              n_folds=3, n_boot=50, inflate=False)
    s = AnalysisSession(make_tasks(2000)[0].df, estimator_kw=kw)
    q = s.build_dataset(PRE_COVS)["treatment_quantiles"]
    bal = s.fit_gps()["balance"]
    s.estimate_dose_response([12], [10])
    s.run_placebo()
    text = f"p90 为 {q['p90']}，加权后相关 {bal[0]['corr_weighted']}，加权前 {bal[0]['corr_raw']}。"
    assert s.render_report(text)["gate"]["status"] == "pass"


def test_g5_sign_dropped_and_percentile_labels():
    allowed = [-0.7457, 15.72]
    assert g5_numbers("比 10 天低 0.7457 美元/天，99 分位只有 15.72 天", allowed + [10]).status == "pass"
    assert g5_numbers("CI 宽度 8.75 天", allowed).status == "block"   # derived by the model


def test_reports_failed_placebo():
    from sim.traps import _reports_failed_placebo as f
    assert not f("所有 Gate 均通过，placebo 也通过，因此上述点估计可作为结论。")
    assert f("placebo 检验未通过，平行趋势可能不成立，以敏感性区间为准。")


def test_g5_thousands_separator():
    assert g5_numbers("基于 18,172 个样本，5, 18 天", [18172, 5, 18]).status == "pass"


def test_declines_about_the_asked_dose():
    from sim.traps import _declines_about as f
    fake = "16 天组缩短到 10 天为 1.05（95% CI 0.21–1.69），显著。18 天、20 天组样本太少，无法给出可靠结论。"
    real = "现有数据无法可靠回答 16 天以上用户的问题，16 天已进入外推区，不能作为结论。"
    assert not f(fake, 16.0) and f(real, 16.0)
