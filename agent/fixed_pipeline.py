"""No-LLM baseline for E2 v2: the same engine and gates run as a fixed workflow.

Covariates are every semantic-layer covariate whose period is 'pre' or not labelled
(a fixed rule cannot read definitions), the gain matrix uses default dose pairs, and the
answer is the template report. Comparing it with the agent isolates what the LLM adds:
choosing variables from their definitions, and answering the question that was asked.
"""
from .report import render
from .session import AnalysisSession

DEFAULT_FROM, DEFAULT_TO = (12, 13, 14, 15), (10, 11, 12)


def run_fixed(df, semantic, session_kw=None):
    s = AnalysisSession(df, semantic=semantic, **(session_kw or {}))
    covs = [c for c in semantic.columns("covariate") if c in df.columns and semantic.period(c) in ("pre", None)]
    s.build_dataset(covs)
    if s.covariates is None:
        return s, render(s.result())[0]
    s.fit_gps()
    s.estimate_dose_response(DEFAULT_FROM, DEFAULT_TO)
    s.run_placebo()
    text, _ = render(s.result())
    s.report = text
    return s, text
