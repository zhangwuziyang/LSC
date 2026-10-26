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


def simulate(n=20000, confounding=1.0, tau=0.0, heavy_tail=True, noise=1.0, dose_model="location_scale",
             interaction=0.0, seed=0):
    """dose_model:
      "location_scale"  log D = m(X) + s(X) eps (the GPS's own form; default, used in E1)
      "bimodal"         two shipping channels (express / standard) whose mix depends on X,
                        so D | X is bimodal and no location-scale model fits it
      "gamma"           D - 2 ~ Gamma with an X-dependent shape, so skewness changes with X
    interaction adds a confounding term zh * zp (and the same covariates drive D more),
    making the outcome model harder to learn."""
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
    if dose_model == "bimodal":
        p_express = 1 / (1 + np.exp(-(-0.3 - 0.9 * zh + 2.0 * reg - 0.4 * zp * confounding)))
        express = rng.uniform(size=n) < p_express
        logd = np.where(express, rng.normal(np.log(6.0) + 0.10 * zh, 0.15, n),
                        rng.normal(np.log(13.0) + 0.20 * zh + 0.08 * u, 0.20, n))
        D = np.clip(np.exp(logd), 2.0, 45.0)
    elif dose_model == "gamma":
        shape = np.exp(0.6 + 0.6 * zp)                       # skewness 2 / sqrt(shape) varies with X
        mean = np.exp(np.log(7.5) + 0.25 * zh + reg + 0.05 * confounding * zp + 0.08 * u)
        D = np.clip(2.0 + rng.gamma(shape, mean / shape), 2.0, 45.0)
    elif dose_model != "location_scale":
        raise ValueError(dose_model)
    if interaction:
        D = np.clip(D * np.exp(0.10 * interaction * zp * zh), 2.0, 45.0)

    tier = np.searchsorted(np.cumsum(TIER_SHARE)[:-1], np.argsort(np.argsort(log_pre)) / n, side="right")
    h = TIER_EFFECT[tier]
    g = confounding * (-0.5 * zh - 0.3 * zh ** 2 + 2.0 * reg + 0.3 * np.tanh(zp) + 0.1 * np.log1p(orders))
    g = g + interaction * 0.6 * zh * zp
    dY = mu_shape(D) * h + g + tau * u + rng.normal(0, noise, n)
    dY_placebo = 0.7 * g + tau * u + rng.normal(0, noise, n)

    X = np.column_stack([np.log(hist), region, log_pre, orders, cat, tenure]).astype(float)
    df = pd.DataFrame(X, columns=COVARIATES).assign(D=D, dY=dY, dY_placebo=dY_placebo, tier=tier)
    return SimData(df=df, X=X, D=D, dY=dY, dY_placebo=dY_placebo, tier=tier,
                   mean_tier_effect=float(h.mean()))
