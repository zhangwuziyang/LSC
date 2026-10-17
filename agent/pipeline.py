"""Deterministic analysis pipeline with gates. The LLM orchestrator calls these steps as
tools; every number it may report comes from the dict returned by `run_analysis`."""
import numpy as np

from gpsdid import (balance_table, dr_dose_response, gain_matrix, grade_overlap, kink,
                    placebo_check, sensitivity)

from .gates import g1_timing, g2_balance, g3_overlap, g4_placebo
from .semantic import SemanticLayer


def run_analysis(df, semantic=None, covariates=None, grid=None, d0=10.0, kink_range=(6, 18),
                 from_days=(12, 13, 14, 15), to_days=(10, 11, 12), days=76,
                 ess_thresholds=(500, 100), estimator_kw=None, seed=0):
    semantic = semantic or SemanticLayer()
    grid = np.arange(5.0, 20.01, 0.5) if grid is None else np.asarray(grid, float)
    covariates = list(covariates or semantic.columns("covariate"))
    treat = semantic.columns("treatment")[0]
    outcome = semantic.columns("outcome")[0]
    placebo = semantic.columns("placebo_outcome")[0]
    kw = dict(seed=seed, **(estimator_kw or {}))
    gates = []

    g1, bad = g1_timing(covariates, semantic)
    gates.append(g1)
    out = dict(treatment=treat, outcome=outcome, covariates=covariates, d0=d0, days=days,
               n=int(len(df)), gates=gates)
    if g1.status == "block":
        out["blocked_columns"] = bad
        return out

    X, D = df[covariates].to_numpy(float), df[treat].to_numpy(float)
    res = dr_dose_response(X, D, df[outcome].to_numpy(float), grid, **kw)
    bal = balance_table(X, D, res.weights, covariates)
    gates.append(g2_balance(bal))

    grades = grade_overlap(res.ess, *ess_thresholds)
    gates.append(g3_overlap(grid, grades))
    usable = grades != "extrapolation"

    pl = dr_dose_response(X, D, df[placebo].to_numpy(float), grid, **kw)
    passed, pl_point, *_ = placebo_check(pl, d0, mask=usable)
    gates.append(g4_placebo(passed, float(np.abs(pl_point[usable]).max())))

    theta, draws = res.contrast(d0)
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    se = draws.std(0)
    _, sens_hi, breakdown = sensitivity(theta, se, pl_point, M=1.0)
    sens_lo, _, _ = sensitivity(theta, se, pl_point, M=1.0)
    k, k_ci = kink(res, *kink_range, seed=seed)
    gm = gain_matrix(res, D, from_days, to_days, days=days, trusted=ess_thresholds[0],
                     caution=ess_thresholds[1])

    out.update(
        placebo_passed=passed,
        curve=dict(d=grid.tolist(), theta=theta.tolist(), lo=lo.tolist(), hi=hi.tolist(),
                   ess=res.ess.tolist(), grade=grades.tolist(),
                   sens_lo=sens_lo.tolist(), sens_hi=sens_hi.tolist(),
                   breakdown_M=np.where(np.isfinite(breakdown), breakdown, -1).tolist()),
        balance=bal.to_dict("records"),
        kink=dict(point=float(k), lo=float(k_ci[0]), hi=float(k_ci[1])),
        gains=gm.to_dict("records"),
        bandwidth=float(res.h),
    )
    return out
