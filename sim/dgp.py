"""Semi-synthetic data with a known dose-response curve.

Mimics the logistics design: D = average delivery days in the treatment window
(right-skewed, heavy-tailed, heteroskedastic), outcomes are daily-average GMV changes,
and the true curve has a kink at 10 days. A latent factor u shifts D and, when tau > 0,
also the outcome trend, which violates conditional parallel trends and shows up in the
placebo outcome.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

TIER_EFFECT = np.array([0.3, 0.6, 1.0, 2.5])       # low / mid / high / top value tiers
TIER_SHARE = np.array([0.35, 0.35, 0.20, 0.10])
COVARIATES = ["log_hist_days", "region", "log_pre_gmv", "orders", "cat_share", "tenure"]


def mu_shape(d, kink=10.0, slope_pre=0.015, slope_post=0.08):
    """Per-unit dose response (tier effect 1): gentle decline, steeper after the kink."""
    d = np.asarray(d, float)
    return -slope_pre * (d - kink) - slope_post * np.maximum(0.0, d - kink)


@dataclass
class SimData:
    df: pd.DataFrame
    X: np.ndarray
    D: np.ndarray
    dY: np.ndarray
    dY_placebo: np.ndarray
    tier: np.ndarray
    mean_tier_effect: float

    def true_mu(self, d):
        """True average dose response up to an additive constant (contrasts are exact)."""
        return mu_shape(d) * self.mean_tier_effect

    def true_contrast(self, d0, d):
        return self.true_mu(d) - self.true_mu(d0)


def _z(v):
    return (v - v.mean()) / v.std()


def simulate(n=20000, confounding=1.0, tau=0.0, heavy_tail=True, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    hist = np.exp(rng.normal(np.log(9.0), 0.35, n))
    region = rng.integers(0, 5, n)
    log_pre = rng.normal(3.0, 1.0, n)
    orders = rng.poisson(np.exp(0.3 * (log_pre - 3.0) + 1.0))
    cat = rng.uniform(0, 1, n)
    tenure = rng.exponential(2.0, n)
    u = rng.normal(0, 1, n)

    zh, zp = _z(np.log(hist)), _z(log_pre)
    reg = np.array([-0.10, 0.0, 0.05, 0.15, 0.30])[region]
    m = np.log(9.0) + 0.25 * zh + reg + 0.08 * np.sin(3 * np.pi * cat) + 0.05 * confounding * zp + 0.08 * u
    s = 0.17 * np.exp(0.30 * zh)
    eps = rng.normal(0, 1, n)
    if heavy_tail:
        tail = rng.uniform(size=n) < 0.15
        eps[tail] = rng.standard_t(3, tail.sum()) / np.sqrt(3) * 1.5
    D = np.clip(np.exp(m + s * eps), 2.0, 45.0)

    tier = np.searchsorted(np.cumsum(TIER_SHARE)[:-1], np.argsort(np.argsort(log_pre)) / n, side="right")
    h = TIER_EFFECT[tier]
    g = confounding * (-0.5 * zh - 0.3 * zh ** 2 + 2.0 * reg + 0.3 * np.tanh(zp) + 0.1 * np.log1p(orders))
    dY = mu_shape(D) * h + g + tau * u + rng.normal(0, noise, n)
    dY_placebo = 0.7 * g + tau * u + rng.normal(0, noise, n)

    X = np.column_stack([np.log(hist), region, log_pre, orders, cat, tenure]).astype(float)
    df = pd.DataFrame(X, columns=COVARIATES).assign(D=D, dY=dY, dY_placebo=dY_placebo, tier=tier)
    return SimData(df=df, X=X, D=D, dY=dY, dY_placebo=dY_placebo, tier=tier,
                   mean_tier_effect=float(h.mean()))
