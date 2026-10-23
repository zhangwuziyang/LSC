import numpy as np
import pytest

from gpsdid import (HeteroGBMGPS, NormalGPS, balance_table, dr_dose_response, gain_matrix,
                    grade_overlap, hirano_imbens, kink, placebo_check, sensitivity)
from gpsdid.inference import local_linear, local_linear_weights, select_bandwidth, sup_t_band
from sim.dgp import simulate

FAST = dict(mean_kw=dict(n_estimators=60, n_jobs=1), scale_kw=dict(n_estimators=40, n_jobs=1))


def fast_gps():
    return HeteroGBMGPS(**FAST)


def fast_outcome(seed=0):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=80, num_leaves=15, n_jobs=1, random_state=seed, verbose=-1)


@pytest.fixture(scope="module")
def sim():
    return simulate(4000, seed=3)


# The normal model puts mass on negative days, so it is integrated over a wider range.
@pytest.mark.parametrize("model,lo", [(NormalGPS, -80.0), (fast_gps, 0.5)])
def test_gps_density_integrates_to_one(sim, model, lo):
    g = model().fit(sim.X, sim.D)
    d = np.linspace(lo, 80, 8000)
    P = g.params(sim.X[:20])
    dens = g.density_from_params(P[:, None, :], d[None, :])
    assert np.allclose(np.trapezoid(dens, d, axis=1), 1.0, atol=0.03)


def test_local_linear_is_exact_for_lines():
    x = np.random.default_rng(0).uniform(0, 10, 500)
    grid = np.linspace(1, 9, 9)
    assert np.allclose(local_linear(x, 2 + 3 * x, grid, h=0.7), 2 + 3 * grid)
    L = local_linear_weights(x, grid, 0.7)
    assert np.allclose(L.sum(0), 1) and np.allclose(L.T @ x, grid)


def test_bandwidth_is_positive_and_finite():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 2000)
    h = select_bandwidth(x, np.sin(x) + rng.normal(0, 0.3, 2000))
    assert np.isfinite(h) and 0.05 < h < 3


def test_sup_t_band_contains_pointwise():
    rng = np.random.default_rng(2)
    point = np.zeros(20)
    draws = rng.normal(size=(400, 20))
    lo, hi = sup_t_band(point, draws)
    plo, phi = np.quantile(draws, [0.025, 0.975], axis=0)
    assert np.all(lo <= plo + 0.05) and np.all(hi >= phi - 0.05)


def test_dr_recovers_contrast(sim):
    grid = np.arange(6, 16.01, 1.0)
    res = dr_dose_response(sim.X, sim.D, sim.dY, grid, gps=fast_gps, outcome=fast_outcome,
                           n_folds=3, n_boot=100, inflate=False)
    p, _ = res.contrast(10)
    truth = sim.true_contrast(10, grid)
    assert np.sqrt(np.mean((p - truth) ** 2)) < 0.15
    assert res.ess.shape == grid.shape and np.all(res.ess <= len(sim.D))


def test_hirano_imbens_runs(sim):
    grid = np.arange(6, 16.01, 1.0)
    res = hirano_imbens(sim.X, sim.D, sim.dY, grid, n_boot=50)
    assert res.draws.shape == (50, len(grid)) and np.all(np.isfinite(res.mu))


def test_kink_and_gains_on_known_curve():
    from gpsdid import DoseResponse
    grid = np.arange(5, 20.01, 0.5)
    mu = -0.02 * grid - 0.1 * np.maximum(0, grid - 11)
    draws = mu + np.random.default_rng(0).normal(0, 0.005, (200, len(grid)))
    D = np.repeat(np.arange(5, 21), 10).astype(float)
    res = DoseResponse(grid=grid, mu=mu, draws=draws, method="test", D=D, ess=np.full(len(grid), 1000.0))
    point, ci = kink(res, 6, 18)
    assert point == pytest.approx(11, abs=0.5)
    gm = gain_matrix(res, D, from_days=[14], to_days=[10, 12], days=10)
    row = gm.set_index("to_day").loc[10]
    assert row.n_units == 10
    assert row.gain_per_day == pytest.approx(10 * (mu[grid == 10][0] - mu[grid == 14][0]) / 10)
    assert set(gm.grade) == {"trusted"}


def test_diagnostics(sim):
    tbl = balance_table(sim.X, sim.D, np.ones_like(sim.D))
    assert np.allclose(tbl.corr_raw, tbl.corr_weighted)
    assert list(grade_overlap([250, 100, 30])) == ["trusted", "caution", "extrapolation"]
    lo, hi, bd = sensitivity(np.array([-1.0]), np.array([0.1]), np.array([0.2]), M=1.0)
    assert lo[0] < -1 < hi[0] and bd[0] == pytest.approx((1 - 0.196) / 0.2)


def test_placebo_detects_trend_violation():
    grid = np.arange(6, 16.01, 1.0)
    kw = dict(gps=fast_gps, outcome=fast_outcome, n_folds=3, n_boot=200, inflate=False)
    clean = simulate(6000, tau=0.0, seed=5)
    bad = simulate(6000, tau=3.0, seed=5)
    _, p_clean, *_ = placebo_check(dr_dose_response(clean.X, clean.D, clean.dY_placebo, grid, **kw), 10)
    ok_bad, p_bad, *_ = placebo_check(dr_dose_response(bad.X, bad.D, bad.dY_placebo, grid, **kw), 10)
    # With these deliberately small nuisance models the clean placebo can still be rejected
    # at the edges (the false-rejection rate is measured in E2), so only compare magnitudes here.
    assert not ok_bad
    assert np.abs(p_bad).max() > 3 * np.abs(p_clean).max()


def test_inflation_widens_draws(sim):
    grid = np.arange(6, 16.01, 1.0)
    kw = dict(gps=fast_gps, outcome=fast_outcome, n_folds=3, n_boot=100, h=1.0, seed=1)
    raw = dr_dose_response(sim.X, sim.D, sim.dY, grid, inflate=False, **kw)
    inf = dr_dose_response(sim.X, sim.D, sim.dY, grid, inflate=True, n_sub=4, **kw)
    assert inf.vif >= 1.0
    assert np.allclose(inf.mu, raw.mu)
    assert np.allclose(inf.draws.std(0), np.sqrt(inf.vif) * raw.draws.std(0))
