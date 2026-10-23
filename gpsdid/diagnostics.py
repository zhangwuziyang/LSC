"""Gate-level diagnostics: covariate balance, overlap grading, placebo, sensitivity."""
import numpy as np
import pandas as pd

from .inference import sup_t_band


def weighted_corr(x, d, w=None):
    w = np.ones_like(d) if w is None else w
    w = w / w.sum()
    xc, dc = x - w @ x, d - w @ d
    return (w @ (xc * dc)) / np.sqrt((w @ (xc * xc)) * (w @ (dc * dc)))


def balance_table(X, D, weights, names=None):
    """Corr(X_k, D) before and after GPS weighting. Under a correct GPS, weighted
    correlations are near zero (balance in the continuous-treatment sense)."""
    X = np.asarray(X, float)
    names = names or [f"x{k}" for k in range(X.shape[1])]
    rows = [(nm, weighted_corr(X[:, k], D), weighted_corr(X[:, k], D, weights))
            for k, nm in enumerate(names)]
    return pd.DataFrame(rows, columns=["covariate", "corr_raw", "corr_weighted"])


def balance_ok(table, threshold=0.1):
    return bool((table["corr_weighted"].abs() < threshold).all())


def grade_overlap(ess, trusted=200, caution=50):
    """Grade each dose by effective sample size of the DR weights.

    Calibrated on E1 (heavy tail, n=20k, variance-inflated CIs): pointwise coverage stays
    near 0.95 at every ESS level, so the grade reports precision, not validity. At
    ESS >= 200 the mean absolute error is <= 0.05 and CIs are ~0.3 wide; below 50 the
    CIs are more than twice as wide, and the GPS model is extrapolating there, which the
    simulation (a correctly specified location-scale GPS) does not stress-test."""
    ess = np.asarray(ess)
    return np.where(ess >= trusted, "trusted", np.where(ess >= caution, "caution", "extrapolation"))


def placebo_check(res_placebo, d0, mask=None, alpha=0.05):
    """Placebo passes if the uniform band of theta_pl(d0, d) covers zero on `mask`."""
    p, dr = res_placebo.contrast(d0)
    lo, hi = sup_t_band(p, dr, alpha)
    m = np.ones_like(p, bool) if mask is None else mask
    return bool(((lo[m] <= 0) & (hi[m] >= 0)).all()), p, lo, hi


def sensitivity(theta, se, theta_placebo, M, z=1.96):
    """Relative-magnitude bound (in the spirit of Rambachan & Roth, 2023): the post-period
    trend violation is at most M times the placebo deviation. Returns the robust CI and the
    breakdown M, the smallest M at which the CI includes zero."""
    bias = M * np.abs(theta_placebo)
    lo, hi = theta - bias - z * se, theta + bias + z * se
    with np.errstate(divide="ignore", invalid="ignore"):
        breakdown = np.maximum(np.abs(theta) - z * se, 0) / np.abs(theta_placebo)
    return lo, hi, breakdown
