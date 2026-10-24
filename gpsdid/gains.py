"""Business outputs: kink location and the gain matrix with bootstrap intervals."""
import numpy as np
import pandas as pd

from .diagnostics import grade_overlap


def _hinge_fit(x, y, candidates, w=None):
    """Best breakpoint c for y ~ a + b*x + g*max(0, x - c) by (weighted) least squares."""
    w = np.ones_like(x) if w is None else w
    best_c, best_sse = None, np.inf
    for c in candidates:
        Z = np.column_stack([np.ones_like(x), x, np.maximum(0, x - c)])
        Zw = Z * w[:, None]
        r = y - Z @ np.linalg.solve(Zw.T @ Z, Zw.T @ y)
        sse = w @ (r * r)
        if sse < best_sse:
            best_c, best_sse = c, sse
    return best_c


def kink(res, lo, hi, step=0.25, n_boot=200, alpha=0.05, seed=0):
    """Breakpoint of a two-segment linear dose response on [lo, hi], with a bootstrap CI.

    With DR pseudo-outcomes the hinge is fit to (D, xi) directly, so no smoothing
    bandwidth blurs the kink; the CI uses Exp(1) multiplier weights. Otherwise it falls
    back to fitting the estimated curve and its bootstrap draws on the grid.
    """
    cand = np.arange(lo + 1.0, hi - 1.0 + 1e-9, step)
    if res.xi is not None:
        m = (res.D >= lo) & (res.D <= hi)
        x, y = res.D[m], res.xi[m]
        rng = np.random.default_rng(seed)
        point = _hinge_fit(x, y, cand)
        if res.groups is None:
            weights = lambda: rng.exponential(1.0, len(x))
        else:  # one weight per group (cluster bootstrap)
            codes, n_groups = res.groups
            weights = lambda: rng.exponential(1.0, n_groups)[codes[m]]
        boot = np.array([_hinge_fit(x, y, cand, weights()) for _ in range(n_boot)])
    else:
        m = (res.grid >= lo) & (res.grid <= hi)
        x = res.grid[m]
        point = _hinge_fit(x, res.mu[m], cand)
        boot = np.array([_hinge_fit(x, row[m], cand) for row in res.draws])
    return point, np.quantile(boot, [alpha / 2, 1 - alpha / 2])


def gain_matrix(res, D, from_days, to_days, days=76, alpha=0.05, trusted=200, caution=50):
    """Daily gain of moving everyone whose rounded dose is `a` to dose `b` (< a):

        Gain(a -> b) = N_a * (mu(b) - mu(a)) / days

    Uses the population curve, i.e. assumes the effect for units at dose a equals the
    average effect; pass a subgroup curve and subgroup D for tier-level matrices.
    Grade = the worse of the two endpoints' overlap grades.
    """
    D = np.asarray(D)
    rounded = np.rint(D)
    order = {"trusted": 0, "caution": 1, "extrapolation": 2}
    grade_at = lambda d: grade_overlap(np.interp([d], res.grid, res.ess), trusted, caution)[0]
    rows = []
    for a in from_days:
        n_a = int((rounded == a).sum())
        for b in to_days:
            if b >= a:
                continue
            theta = res.at(b)[0] - res.at(a)[0]
            th_d = res.at(b, draws=True)[:, 0] - res.at(a, draws=True)[:, 0]
            lo, hi = np.quantile(th_d, [alpha / 2, 1 - alpha / 2])
            ga, gb = grade_at(a), grade_at(b)
            rows.append(dict(from_day=a, to_day=b, n_units=n_a, theta=theta,
                             gain_per_day=n_a * theta / days,
                             gain_lo=n_a * lo / days, gain_hi=n_a * hi / days,
                             grade=max(ga, gb, key=order.get)))
    return pd.DataFrame(rows)
