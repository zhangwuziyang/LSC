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
