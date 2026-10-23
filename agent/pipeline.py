"""Deterministic gated pipeline: the agent's tool steps called in a fixed order.
Every number a report may contain comes from the dict this returns."""
from .session import AnalysisSession


def run_analysis(df, semantic=None, covariates=None, from_days=(12, 13, 14, 15),
                 to_days=(10, 11, 12), **session_kw):
    s = AnalysisSession(df, semantic=semantic, **session_kw)
    covariates = list(covariates or s.semantic.columns("covariate"))
    built = s.build_dataset(covariates)
    if s.covariates is None:
        return {**s.result(), "covariates": covariates,
                "blocked_columns": built.get("blocked_columns", [])}
    s.fit_gps()
    s.estimate_dose_response(from_days, to_days)
    s.run_placebo()
    return s.result()
