"""Stateful analysis session. Each public step is one agent tool; `run_analysis` in
pipeline.py calls the same steps in order, so the deterministic pipeline and the LLM
agent share one code path. Every step returns a JSON-serializable dict."""
import numpy as np

from gpsdid import (balance_table, dr_dose_response, gain_matrix, grade_overlap, kink,
                    placebo_check, sensitivity)

from .gates import g1_timing, g2_balance, g3_overlap, g4_placebo
from .report import render
from .semantic import SemanticLayer


def _r(x, k=4):
    return float(round(float(x), k))


class StepError(Exception):
    """A step was called before the steps it depends on."""


class AnalysisSession:
    def __init__(self, df, semantic=None, grid=None, d0=10.0, days=76, kink_range=(6, 18),
                 ess_thresholds=(200, 50), estimator_kw=None, gates_enabled=True,
                 semantic_enabled=True, seed=0):
        """gates_enabled / semantic_enabled switch off parts of the system for the E2
        ablations. Gates are still computed and kept in self.gates for grading, but with
        gates off the model is told every gate passed and nothing is blocked; with the
        semantic layer off the model sees column names only."""
        self.gates_enabled = gates_enabled
        self.semantic_enabled = semantic_enabled
        self.df = df
        self.semantic = semantic or SemanticLayer()
        self.grid = np.arange(5.0, 20.01, 0.5) if grid is None else np.asarray(grid, float)
        self.d0, self.days, self.kink_range = d0, days, kink_range
        self.ess_thresholds = ess_thresholds
        self.kw = dict(seed=seed, **(estimator_kw or {}))
        self.seed = seed
        self.treat = self.semantic.columns("treatment")[0]
        self.outcome = self.semantic.columns("outcome")[0]
        self.placebo = self.semantic.columns("placebo_outcome")[0]
        self.gates = {}
        self.covariates = None
        self.res = self.pl_point = None
        self.est = self.pl = None
        self.report = None

    # ---- tools -------------------------------------------------------------------------
    def lookup_metric(self, query):
        if not self.semantic_enabled:
            return {"results": [{"column": c} for c in self.df.columns if query.lower() in c.lower()]}
        hits = self.semantic.lookup(query) or [m for m in self.semantic.metrics if m["role"] == query]
        keys = ("column", "name", "role", "period", "definition", "unit")
        return {"results": [{k: m[k] for k in keys if k in m} for m in hits]}

    def list_metrics(self):
        if not self.semantic_enabled:
            return {"metrics": [{"column": c} for c in self.df.columns]}
        return {"metrics": [{k: m.get(k) for k in ("column", "name", "role", "period")}
                            for m in self.semantic.metrics]}

    def build_dataset(self, covariates):
        g1, bad = g1_timing(list(covariates), self.semantic)
        self._set_gate(g1)
        missing = [c for c in covariates if c not in self.df.columns]
        if (g1.status == "block" and self.gates_enabled) or missing:
            self.covariates = None
            out = {"gate": self._show(g1), "blocked_columns": bad}
            if missing:
                out["missing_columns"] = missing
            return out
        self.covariates = list(covariates)
        self.res = self.est = self.pl = self.report = None
        D = self.df[self.treat].to_numpy(float)
        q = np.quantile(D, [0.1, 0.5, 0.9, 0.99])
        return {"gate": self._show(g1), "n": int(len(D)), "covariates": self.covariates,
                "treatment_quantiles": {"p10": _r(q[0], 2), "p50": _r(q[1], 2),
                                        "p90": _r(q[2], 2), "p99": _r(q[3], 2)}}

    def fit_gps(self):
        if self.covariates is None:
            raise StepError("call build_dataset with valid covariates first")
        X, D = self._xd()
        self.res = dr_dose_response(X, D, self.df[self.outcome].to_numpy(float), self.grid, **self.kw)
        bal = balance_table(X, D, self.res.weights, self.covariates)
        g2 = g2_balance(bal)
        self._set_gate(g2)
        return {"gate": self._show(g2),
                "balance": [{"covariate": r.covariate, "corr_raw": _r(r.corr_raw, 3),
                             "corr_weighted": _r(r.corr_weighted, 3)} for r in bal.itertuples()]}

    def estimate_dose_response(self, from_days=(12, 13, 14, 15), to_days=(10, 11, 12)):
        if self.res is None:
            raise StepError("call fit_gps first")
        res, D = self.res, self._xd()[1]
        grades = grade_overlap(res.ess, *self.ess_thresholds)
        g3 = g3_overlap(self.grid, grades)
        self._set_gate(g3)
        theta, draws = res.contrast(self.d0)
        lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
        k, k_ci = kink(res, *self.kink_range, seed=self.seed)
        gm = gain_matrix(res, D, list(from_days), list(to_days), days=self.days,
                         trusted=self.ess_thresholds[0], caution=self.ess_thresholds[1])
        self.est = dict(theta=theta, lo=lo, hi=hi, se=draws.std(0), grades=grades,
                        kink=(float(k), float(k_ci[0]), float(k_ci[1])), gains=gm)
        ints = [i for i, d in enumerate(self.grid) if d % 1 == 0]
        return {"gate": self._show(g3), "reference_day": self.d0,
                "curve": [{"d": float(self.grid[i]), "theta": _r(theta[i]), "lo": _r(lo[i]),
                           "hi": _r(hi[i]), **({"grade": str(grades[i])} if self.gates_enabled else {})}
                          for i in ints],
                "kink": {"point": _r(k, 2), "lo": _r(k_ci[0], 2), "hi": _r(k_ci[1], 2)},
                "gains_per_day": [{k2: (_r(v, 4) if isinstance(v, float) else v) for k2, v in row.items()
                                   if self.gates_enabled or k2 != "grade"}
                                  for row in gm.to_dict("records")]}

    def run_placebo(self):
        if self.est is None:
            raise StepError("call estimate_dose_response first")
        X, D = self._xd()
        usable = self.est["grades"] != "extrapolation"
        pl = dr_dose_response(X, D, self.df[self.placebo].to_numpy(float), self.grid, **self.kw)
        passed, pl_point, *_ = placebo_check(pl, self.d0, mask=usable)
        g4 = g4_placebo(passed, float(np.abs(pl_point[usable]).max()))
        self._set_gate(g4)
        s_lo, s_hi, bd = sensitivity(self.est["theta"], self.est["se"], pl_point, M=1.0)
        self.pl = dict(passed=passed, point=pl_point, sens_lo=s_lo, sens_hi=s_hi, breakdown=bd)
        ints = [i for i, d in enumerate(self.grid) if d % 1 == 0 and d != self.d0]
        return {"gate": self._show(g4), "placebo_passed": passed or not self.gates_enabled,
                "sensitivity_M1": [{"d": float(self.grid[i]), "lo": _r(s_lo[i]), "hi": _r(s_hi[i]),
                                    "breakdown_M": _r(bd[i], 3) if np.isfinite(bd[i]) else None}
                                   for i in ints]}

    def render_report(self, narrative=""):
        if self.pl is None:
            raise StepError("call run_placebo first")
        result = self.result()
        if not self.gates_enabled:  # the model was told everything passed; so is the report
            result = {**result, "placebo_passed": True,
                      "gates": [type(g)(g.gate, "pass", "（检查已关闭）") for g in result["gates"]],
                      "curve": {**result["curve"], "grade": ["trusted"] * len(result["curve"]["grade"])},
                      "gains": [{**g, "grade": "trusted"} for g in result["gains"]]}
        text, g5 = render(result, narrative)
        self._set_gate(g5)
        ok = g5.status == "pass" or not self.gates_enabled
        self.report = text if ok else None
        out = {"gate": self._show(g5)}
        if ok:
            out["report"] = text
        return out

    # ---- state ------------------------------------------------------------------------
    def _xd(self):
        return self.df[self.covariates].to_numpy(float), self.df[self.treat].to_numpy(float)

    def _show(self, g):
        return g.to_dict() if self.gates_enabled else {"gate": g.gate, "status": "pass", "detail": "（检查已关闭）"}

    def _set_gate(self, g):
        self.gates[g.gate] = g

    def result(self):
        """Engine output in the shape report.render expects."""
        out = dict(treatment=self.treat, outcome=self.outcome, covariates=self.covariates,
                   d0=self.d0, days=self.days, n=int(len(self.df)),
                   gates=[g for name, g in self.gates.items() if not name.startswith("G5")])
        if self.est is None:
            return out
        e = self.est
        out.update(curve=dict(d=self.grid.tolist(), theta=e["theta"].tolist(), lo=e["lo"].tolist(),
                              hi=e["hi"].tolist(), ess=self.res.ess.tolist(), grade=e["grades"].tolist()),
                   balance=[], kink=dict(point=e["kink"][0], lo=e["kink"][1], hi=e["kink"][2]),
                   gains=e["gains"].to_dict("records"), bandwidth=float(self.res.h))
        if self.pl is not None:
            p = self.pl
            out["placebo_passed"] = p["passed"]
            out["curve"].update(sens_lo=p["sens_lo"].tolist(), sens_hi=p["sens_hi"].tolist(),
                                breakdown_M=np.where(np.isfinite(p["breakdown"]), p["breakdown"], -1).tolist())
        return out
