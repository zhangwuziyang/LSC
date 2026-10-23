"""E1: GPS model (M0 normal / M1 hetero-GBM) x estimator (E0 Hirano-Imbens / E1 doubly robust).

Target: the contrast theta(10, d) = mu(d) - mu(10) on d in [6, 16], against the known truth
of the semi-synthetic DGP. Writes per-replication metrics, per-grid estimates and a summary.

    python experiments/e1_estimators.py --reps 20 --n 20000
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gpsdid import HeteroGBMGPS, NormalGPS, dr_dose_response, hirano_imbens, kink
from gpsdid.inference import sup_t_band
from sim.dgp import simulate

GRID = np.arange(5.0, 20.01, 0.5)
D0 = 10.0
EVAL = (GRID >= 6) & (GRID <= 16)
METHODS = {
    "E0+M0 (original)": lambda s, seed: hirano_imbens(s.X, s.D, s.dY, GRID, gps=NormalGPS, seed=seed),
    "E0+M1": lambda s, seed: hirano_imbens(s.X, s.D, s.dY, GRID, gps=HeteroGBMGPS, seed=seed),
    "E1+M0": lambda s, seed: dr_dose_response(s.X, s.D, s.dY, GRID, gps=NormalGPS, seed=seed),
    "E1+M1 (proposed)": lambda s, seed: dr_dose_response(s.X, s.D, s.dY, GRID, gps=HeteroGBMGPS, seed=seed),
}
SCENARIOS = {"heavy_tail": dict(heavy_tail=True), "normal_tail": dict(heavy_tail=False)}


def evaluate(res, truth, true_kink=10.0):
    p, dr = res.contrast(D0)
    lo, hi = np.quantile(dr, [0.025, 0.975], axis=0)
    blo, bhi = sup_t_band(p, dr)
    e = EVAL & (GRID != D0)
    k, k_ci = kink(res, 6, 18)
    return dict(
        rmse=np.sqrt(np.mean((p - truth)[e] ** 2)),
        abs_bias=np.mean(np.abs(p - truth)[e]),
        coverage=np.mean(((lo <= truth) & (truth <= hi))[e]),
        band_covers=float(np.all((blo <= truth)[e] & (truth <= bhi)[e])),
        ci_width=np.mean((hi - lo)[e]),
        kink_err=abs(k - true_kink),
        kink_covers=float(k_ci[0] <= true_kink <= k_ci[1]),
        kink_ci_width=float(k_ci[1] - k_ci[0]),
    ), dict(est=p, lo=lo, hi=hi, ess=res.ess if res.ess is not None else np.full_like(p, np.nan))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--scenario", choices=list(SCENARIOS), action="append",
                    help="run only these scenarios (default: all); results go to e1_*_<scenario> files")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scenarios = args.scenario or list(SCENARIOS)
    tag = "" if args.scenario is None else "_" + "_".join(scenarios)
    rows, curves = [], []
    for scen in scenarios:
        kw = SCENARIOS[scen]
        for rep in range(args.reps):
            s = simulate(args.n, seed=rep, **kw)
            truth = s.true_contrast(D0, GRID)
            for name, fn in METHODS.items():
                t = time.time()
                met, cur = evaluate(fn(s, rep), truth)
                rows.append(dict(scenario=scen, rep=rep, method=name, seconds=time.time() - t, **met))
                curves.append(pd.DataFrame(dict(scenario=scen, rep=rep, method=name, d=GRID,
                                                truth=truth, **cur)))
            print(f"{scen} rep {rep + 1}/{args.reps} done", flush=True)
            pd.DataFrame(rows).to_csv(out / f"e1_metrics{tag}.csv", index=False)
            pd.concat(curves).round(5).to_csv(out / f"e1_curves{tag}.csv", index=False)

    df = pd.DataFrame(rows)
    summary = (df.groupby(["scenario", "method"], sort=False)
                 [["rmse", "abs_bias", "coverage", "band_covers", "ci_width", "kink_err", "kink_covers",
                   "kink_ci_width", "seconds"]]
                 .mean().round(3))
    summary.to_csv(out / f"e1_summary{tag}.csv")
    (out / f"e1_summary{tag}.md").write_text(
        f"# E1 summary (n={args.n}, reps={args.reps})\n\n"
        "Target: theta(10, d) on d in [6, 16]. coverage = mean pointwise 95% CI coverage; "
        "band_covers = share of reps whose 95% uniform band covers the whole true curve; "
        "kink_err = |estimated kink - 10| in days; kink_covers = share of reps whose 95% kink CI "
        "contains 10.\n\n" + summary.to_markdown() + "\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()
