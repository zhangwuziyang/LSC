"""Dose-response estimators for ΔY = Y_post - Y_pre on a continuous dose D.

E0  hirano_imbens       two-step GPS regression (baseline; the original business method)
E1  dr_dose_response    doubly robust pseudo-outcome + local linear (Kennedy et al., 2017),
                        with cross-fitting
"""
from dataclasses import dataclass, field

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

from .gps import HeteroGBMGPS, NormalGPS
from .inference import (local_linear, local_linear_multiplier_bootstrap,
                        local_linear_weights, select_bandwidth, sup_t_band)


@dataclass
class DoseResponse:
    grid: np.ndarray
    mu: np.ndarray            # point estimate of mu(d) on grid
    draws: np.ndarray         # bootstrap draws, shape (B, G)
    method: str
    D: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False, default=None)   # unclipped varpi(D)/pi(D|X), for balance
    h: float = None
    ess: np.ndarray = None    # effective sample size at each grid point
    xi: np.ndarray = field(repr=False, default=None)        # DR pseudo-outcomes
    vif: float = 1.0          # variance inflation applied to the draws (see subsample_inflation)

    def ci(self, alpha=0.05):
        lo, hi = np.quantile(self.draws, [alpha / 2, 1 - alpha / 2], axis=0)
        return lo, hi

    def band(self, alpha=0.05):
        return sup_t_band(self.mu, self.draws, alpha)

    def at(self, d, draws=False):
        """mu at arbitrary doses by linear interpolation of the grid (point or draws)."""
        d = np.atleast_1d(np.asarray(d, float))
        if not draws:
            return np.interp(d, self.grid, self.mu)
        return np.stack([np.interp(d, self.grid, row) for row in self.draws])

    def contrast(self, d0):
        """theta(d0, d) = mu(d) - mu(d0) on the grid: (point, draws)."""
        p = self.mu - self.at(d0)[0]
        dr = self.draws - self.at(d0, draws=True)[:, :1]
        return p, dr


def default_outcome_model(seed=0):
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                         min_child_samples=50, random_state=seed, verbose=-1)


def _ess(D, grid, h, w):
    """ESS of the effective weights a_i(d) = l_i(d) * w_i that multiply ΔY at dose d."""
    A = local_linear_weights(D, grid, h) * w[:, None]
    return A.sum(0) ** 2 / (A * A).sum(0)


def _crossfit(X, D, dY, gps, outcome, n_folds, n_mc, dgrid, seed):
    """One cross-fitting pass: per-unit residual, stabilized weight and mean outcome."""
    n = len(D)
    rng = np.random.default_rng(seed)
    resid, ratio, mbar = np.empty(n), np.empty(n), np.empty(n)
    for k, (tr, te) in enumerate(KFold(n_folds, shuffle=True, random_state=seed).split(X)):
        g = gps().fit(X[tr], D[tr])
        om = outcome(seed + k).fit(np.column_stack([X[tr], D[tr]]), dY[tr])
        pi = g.density_from_params(g.params(X[te]), D[te])
        mc = rng.choice(te, min(n_mc, len(te)), replace=False)
        P_mc = g.params(X[mc])
        varpi = g.density_from_params(P_mc[:, None, :], dgrid[None, :]).mean(0)
        Xg = np.column_stack([np.repeat(X[mc], len(dgrid), 0), np.tile(dgrid, len(mc))])
        mbar_grid = om.predict(Xg).reshape(len(mc), len(dgrid)).mean(0)
        resid[te] = dY[te] - om.predict(np.column_stack([X[te], D[te]]))
        ratio[te] = np.interp(D[te], dgrid, varpi) / np.maximum(pi, 1e-12)
        mbar[te] = np.interp(D[te], dgrid, mbar_grid)
    return resid, ratio, mbar


def dr_dose_response(X, D, dY, grid, gps=HeteroGBMGPS, outcome=default_outcome_model,
                     n_folds=5, n_splits=1, h=None, undersmooth=0.5, n_boot=500,
                     weight_clip=0.999, n_mc=1000, n_dose_grid=150, inflate=True,
                     n_sub=12, sub_frac=0.5, seed=0):
    """Cross-fitted doubly robust dose-response curve.

    Pseudo-outcome (Kennedy et al., 2017):
        xi_i = (ΔY_i - m(X_i, D_i)) * varpi(D_i) / pi(D_i | X_i) + mean_j m(X_j, D_i)
        varpi(d) = mean_j pi(d | X_j)
    mu(d) is the local linear regression of xi on D. The two marginal integrals are
    computed on a dose grid from a Monte Carlo subsample of n_mc units per fold and
    interpolated. Weights above the `weight_clip` quantile are capped inside xi only;
    the returned `weights` are unclipped, since capping itself breaks balance.

    Without an explicit `h`, the bandwidth is `undersmooth` x the CV-optimal one. CV picks
    a bandwidth that is MSE-optimal for the curve, and its smoothing bias (largest at the
    kink) makes bootstrap intervals undercover; undersmoothing trades width for coverage.

    With n_splits > 1 the cross-fitting is repeated on independent fold splits
    (Chernozhukov et al., 2018). The point estimate uses the split-averaged xi, and the
    bootstrap draws of all splits are pooled, so their spread includes the variation
    across splits as well as the within-split sampling variance. Each split's xi is
    noisier than the average, so the pooled spread is slightly conservative.

    The multiplier bootstrap treats xi as fixed and so misses the variance of the
    nuisance fits; in simulations its SE is ~20% too small and pointwise coverage ~0.90.
    With inflate=True (default) the draws are rescaled by a variance inflation factor
    estimated by subsampling (see `subsample_inflation`); half-size subsamples gave
    coverage 0.97 (quarter-size: 0.99, too conservative). Costs ~n_sub * sub_frac extra fits.

    `gps` and `outcome` are factories (called with no args / with seed) so each fold
    gets fresh models.
    """
    X = np.asarray(X, float)
    D = np.asarray(D, float)
    dY = np.asarray(dY, float)
    grid = np.asarray(grid, float)
    dgrid = np.linspace(*np.quantile(D, [0.001, 0.999]), n_dose_grid)
    xis, ratios = [], []
    for sp in range(n_splits):
        resid, ratio, mbar = _crossfit(X, D, dY, gps, outcome, n_folds, n_mc, dgrid, seed + 1000 * sp)
        w_used = np.minimum(ratio, np.quantile(ratio, weight_clip))
        xis.append(resid * w_used + mbar)
        ratios.append(ratio)
    xi = np.mean(xis, 0)
    ratio = np.mean(ratios, 0)
    w_used = np.minimum(ratio, np.quantile(ratio, weight_clip))
    if h is None:
        h = undersmooth * select_bandwidth(D, xi, seed=seed)
    mu = local_linear(D, xi, grid, h)
    # Each split's draws are centred on that split's own curve, so pooling them carries
    # the between-split variation.
    per = max(1, n_boot // n_splits)
    draws = np.concatenate([local_linear_multiplier_bootstrap(D, x_s, grid, h, n_boot=per, seed=seed + sp)
                            for sp, x_s in enumerate(xis)])
    res = DoseResponse(grid=grid, mu=mu, draws=draws,
                       method=f"DR[{gps.__name__ if hasattr(gps, '__name__') else 'gps'}]",
                       D=D, weights=ratio, h=h, ess=_ess(D, grid, h, w_used), xi=xi)
    if inflate:
        kw = dict(gps=gps, outcome=outcome, n_folds=n_folds, weight_clip=weight_clip,
                  n_mc=n_mc, n_dose_grid=n_dose_grid)
        res.vif = subsample_inflation(X, D, dY, res, kw, n_sub=n_sub, frac=sub_frac, seed=seed)
        res.draws = mu + np.sqrt(res.vif) * (draws - mu)
    return res


def subsample_inflation(X, D, dY, res, fit_kw, n_sub=12, frac=0.5, seed=0):
    """Scalar variance inflation factor for the multiplier-bootstrap draws.

    Refits the full estimator (nuisances included, bandwidth held at res.h) on n_sub
    subsamples of size b = frac * n drawn without replacement. For subsamples of one
    data set, Var(mu_b) = sigma^2 (1/b - 1/n), so the full-sample variance is
    Var(mu_b) / (n/b - 1). Nuisances fit on b < n points are worse, which makes the
    factor conservative. The factor is the median ratio over grid points (one scalar
    keeps the correlation structure of the draws), and never below 1.
    """
    rng = np.random.default_rng(seed + 7919)
    n = len(D)
    b = int(frac * n)
    sub = np.stack([
        dr_dose_response(X[idx], D[idx], dY[idx], res.grid, h=res.h, n_boot=1, inflate=False,
                         seed=seed + j + 1, **fit_kw).mu
        for j, idx in enumerate(rng.choice(n, b, replace=False) for _ in range(n_sub))])
    sub_var = sub.var(0, ddof=1) / (n / b - 1)
    ratio = sub_var / res.draws.var(0)
    return float(max(1.0, np.median(ratio[np.isfinite(ratio)])))


def _hi_design(d, r):
    return np.stack([np.ones_like(d), d, d * d, r, r * r, d * r], axis=-1)


def hirano_imbens(X, D, dY, grid, gps=NormalGPS, n_boot=500, n_mc=2000, seed=0):
    """E0: Hirano & Imbens (2004) two-step estimator.

    1) R_i = pi(D_i | X_i);  2) ΔY ~ [1, D, D², R, R², D·R] by OLS;
    3) mu(d) = mean_j beta' [1, d, d², r_j(d), r_j(d)², d·r_j(d)].
    The bootstrap reweights step 2 with Exp(1) multipliers, holding the GPS fixed.
    """
    X = np.asarray(X, float)
    D = np.asarray(D, float)
    dY = np.asarray(dY, float)
    grid = np.asarray(grid, float)
    rng = np.random.default_rng(seed)
    g = gps().fit(X, D)
    P = g.params(X)
    R = g.density_from_params(P, D)
    Z = _hi_design(D, R)
    beta = np.linalg.lstsq(Z, dY, rcond=None)[0]
    mc = rng.choice(len(D), min(n_mc, len(D)), replace=False)
    Rg = g.density_from_params(P[mc][:, None, :], grid[None, :])          # (M, G)
    Zg = _hi_design(np.broadcast_to(grid, Rg.shape), Rg).mean(0)          # (G, 6)
    mu = Zg @ beta
    draws = np.empty((n_boot, len(grid)))
    for b in range(n_boot):
        v = rng.exponential(1.0, len(D))
        Zv = Z * v[:, None]
        draws[b] = Zg @ np.linalg.solve(Zv.T @ Z, Zv.T @ dY)
    f_d = np.interp(D, *_marginal_density(D))
    weights = f_d / np.maximum(R, 1e-12)
    return DoseResponse(grid=grid, mu=mu, draws=draws, method=f"HI[{gps.__name__}]",
                        D=D, weights=weights)


def _marginal_density(D, n_grid=512):
    sd = min(D.std(), np.subtract(*np.percentile(D, [75, 25])) / 1.34)
    bw = 0.9 * sd * len(D) ** -0.2
    xs = np.linspace(D.min(), D.max(), n_grid)
    sub = D if len(D) <= 20000 else np.random.default_rng(0).choice(D, 20000, replace=False)
    dens = np.exp(-0.5 * ((xs[:, None] - sub[None, :]) / bw) ** 2).sum(1)
    return xs, dens / np.trapezoid(dens, xs)
