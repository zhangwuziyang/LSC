"""Generalized propensity score (GPS) models: conditional densities pi(d | x).

Every model exposes the same interface:

    fit(X, D)                    -> self
    params(X)                    -> array (n, k) of per-unit density parameters
    density_from_params(P, d)    -> pi(d | x) for parameters P (..., k), d broadcastable to P.shape[:-1]
    density(X, d)                -> shortcut for density_from_params(params(X), d)

Splitting params() from the density lets estimators evaluate pi(d | X_j) on a whole
dose grid with one model prediction per unit.
"""
import numpy as np
from scipy import stats
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

_SQRT2PI = np.sqrt(2.0 * np.pi)


class NormalGPS:
    """M0: D | X ~ N(X beta, sigma^2), the Hirano & Imbens (2004) specification."""

    def fit(self, X, D):
        Z = np.column_stack([np.ones(len(X)), X])
        self.beta_, *_ = np.linalg.lstsq(Z, D, rcond=None)
        r = D - Z @ self.beta_
        self.sigma_ = np.sqrt(r @ r / (len(D) - Z.shape[1]))
        return self

    def params(self, X):
        loc = np.column_stack([np.ones(len(X)), X]) @ self.beta_
        return loc[:, None]

    def density_from_params(self, P, d):
        z = (d - P[..., 0]) / self.sigma_
        return np.exp(-0.5 * z * z) / (self.sigma_ * _SQRT2PI)

    def density(self, X, d):
        return self.density_from_params(self.params(X), d)


class HeteroGBMGPS:
    """M1: location-scale model on log D with gradient boosting and a nonparametric error.

        log D = m(X) + s(X) * eps,   eps ~ f (kernel density of standardized residuals)
        pi(d | x) = f((log d - m(x)) / s(x)) / (s(x) * d)

    m is fit by LightGBM; s is fit by LightGBM on log squared out-of-fold residuals.
    Any constant bias in the log-variance fit is absorbed by f.
    """

    # log squared residuals are very noisy, so the scale model is regularized harder.
    MEAN_KW = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=100)
    SCALE_KW = dict(n_estimators=150, learning_rate=0.05, num_leaves=7, min_child_samples=200)

    def __init__(self, log_transform=True, mean_kw=None, scale_kw=None, n_oof=2,
                 kde_max=20000, kde_grid=2048, tail_weight=0.01, seed=0):
        self.log_transform = log_transform
        base = dict(random_state=seed, verbose=-1)
        self.mean_kw = {**self.MEAN_KW, **(mean_kw or {}), **base}
        self.scale_kw = {**self.SCALE_KW, **(scale_kw or {}), **base}
        self.n_oof = n_oof
        self.kde_max = kde_max
        self.kde_grid = kde_grid
        self.tail_weight = tail_weight
        self.seed = seed

    def _t(self, d):
        return np.log(d) if self.log_transform else d

    def fit(self, X, D):
        X = np.asarray(X, float)
        T = self._t(np.asarray(D, float))
        oof = np.empty_like(T)
        for tr, te in KFold(self.n_oof, shuffle=True, random_state=self.seed).split(X):
            oof[te] = LGBMRegressor(**self.mean_kw).fit(X[tr], T[tr]).predict(X[te])
        r = T - oof
        self.mean_ = LGBMRegressor(**self.mean_kw).fit(X, T)
        self.scale_ = LGBMRegressor(**self.scale_kw).fit(X, np.log(r * r + 1e-10))
        eps = r / np.exp(0.5 * self.scale_.predict(X))
        self._fit_kde(eps)
        return self

    def _fit_kde(self, eps):
        rng = np.random.default_rng(self.seed)
        if len(eps) > self.kde_max:
            eps = rng.choice(eps, self.kde_max, replace=False)
        sd = eps.std()
        iqr = np.subtract(*np.percentile(eps, [75, 25]))
        bw = 0.9 * min(sd, iqr / 1.34) * len(eps) ** -0.2
        grid = np.linspace(eps.min() - 4 * bw, eps.max() + 4 * bw, self.kde_grid)
        dens = np.zeros_like(grid)
        for chunk in np.array_split(eps, max(1, len(eps) // 2000)):
            u = (grid[:, None] - chunk[None, :]) / bw
            dens += np.exp(-0.5 * u * u).sum(1)
        dens /= np.trapezoid(dens, grid)
        self.eps_grid_, self.eps_dens_ = grid, dens
        self.eps_sd_ = sd

    def params(self, X):
        X = np.asarray(X, float)
        m = self.mean_.predict(X)
        s = np.exp(0.5 * self.scale_.predict(X))
        return np.column_stack([m, s])

    def density_from_params(self, P, d):
        d = np.asarray(d, float)
        m, s = P[..., 0], P[..., 1]
        e = (self._t(d) - m) / s
        f = np.interp(e, self.eps_grid_, self.eps_dens_, left=0.0, right=0.0)
        # A small Student-t(3) component keeps the density positive beyond the KDE support.
        w = self.tail_weight
        f = ((1 - w) * f + w * stats.t.pdf(e / self.eps_sd_, 3) / self.eps_sd_) / s
        return f / d if self.log_transform else f

    def density(self, X, d):
        return self.density_from_params(self.params(X), d)
