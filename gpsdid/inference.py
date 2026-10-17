"""Local linear smoothing, bandwidth selection, and multiplier-bootstrap bands."""
import numpy as np


def _kernel_terms(x, grid, h):
    U = x[:, None] - grid[None, :]
    K = np.exp(-0.5 * (U / h) ** 2)
    return U, K


def local_linear(x, y, grid, h, w=None):
    """Gaussian-kernel local linear regression of y on x, evaluated at each grid point."""
    U, K = _kernel_terms(x, grid, h)
    if w is not None:
        K = K * w[:, None]
    KU = K * U
    S0, S1, S2 = K.sum(0), KU.sum(0), (KU * U).sum(0)
    T0, T1 = K.T @ y, KU.T @ y
    return (S2 * T0 - S1 * T1) / (S0 * S2 - S1 ** 2)


def local_linear_weights(x, grid, h):
    """Equivalent-kernel weights l_i(d), shape (n, G); each column sums to 1."""
    U, K = _kernel_terms(x, grid, h)
    KU = K * U
    S1, S2 = KU.sum(0), (KU * U).sum(0)
    L = K * S2[None, :] - KU * S1[None, :]
    return L / L.sum(0, keepdims=True)


def local_linear_multiplier_bootstrap(x, y, grid, h, n_boot=500, seed=0, chunk=50):
    """Draws of the local linear fit under Exp(1) multiplier weights, shape (n_boot, G).

    All five kernel sums are linear in the weights, so each chunk of draws is a matrix
    product V @ (n, G) instead of a loop over units.
    """
    rng = np.random.default_rng(seed)
    U, K = _kernel_terms(x, grid, h)
    KU = K * U
    A = [K, KU, KU * U, K * y[:, None], KU * y[:, None]]
    out = np.empty((n_boot, len(grid)))
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        V = rng.exponential(1.0, size=(b, len(x)))
        S0, S1, S2, T0, T1 = (V @ a for a in A)
        out[start:start + b] = (S2 * T0 - S1 * T1) / (S0 * S2 - S1 ** 2)
    return out


def select_bandwidth(x, y, n_candidates=12, max_n=3000, seed=0):
    """Leave-one-out CV for the local linear bandwidth on a subsample, rescaled to full n."""
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.choice(n, min(n, max_n), replace=False)
    xs, ys = x[idx], y[idx]
    m = len(xs)
    sd = min(xs.std(), np.subtract(*np.percentile(xs, [75, 25])) / 1.34)
    base = 1.06 * sd * m ** -0.2
    best_h, best_cv = None, np.inf
    for h in base * np.geomspace(0.3, 3.0, n_candidates):
        U, K = _kernel_terms(xs, xs, h)
        KU = K * U
        S0, S1, S2 = K.sum(0), KU.sum(0), (KU * U).sum(0)
        T0, T1 = K.T @ ys, KU.T @ ys
        den = S0 * S2 - S1 ** 2
        fit = (S2 * T0 - S1 * T1) / den
        lev = S2 / den  # l_jj: K_jj = 1 and U_jj = 0
        if not np.all(np.isfinite(lev)) or lev.max() > 0.99:
            continue
        cv = np.mean(((ys - fit) / (1 - lev)) ** 2)
        if cv < best_cv:
            best_h, best_cv = h, cv
    return best_h * (m / n) ** 0.2


def sup_t_band(point, draws, alpha=0.05):
    """Uniform (sup-t) band from bootstrap draws. Grid points with zero spread are skipped."""
    se = draws.std(0)
    ok = se > 1e-12 * max(1.0, np.abs(point).max())
    t = np.abs(draws[:, ok] - point[ok]) / se[ok]
    c = np.quantile(t.max(1), 1 - alpha)
    return point - c * se, point + c * se
