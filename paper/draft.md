# Assumption-Gated LLM Agents for Continuous-Treatment Difference-in-Differences: Evidence from Cross-Border E-commerce Logistics

*Draft v0.2 — 2026-10-24. All sections drafted; Qwen results beyond the full-system condition pending.*

---

## Abstract

How much does a slower delivery cost an e-commerce platform in future sales? Delivery time is a continuous treatment, so the binary toolkit of propensity score matching does not apply, and practitioners turn to generalized propensity scores (GPS) combined with difference-in-differences (DID). We show that the variant commonly used in industry — a normal GPS with the Hirano–Imbens two-step estimator — is biased by a factor of 2.5–3 and produces confidence intervals that cover the truth in only 12–14% of simulated replications when delivery times are right-skewed and heavy-tailed, as they are in practice. We replace it with a cross-fitted doubly robust dose-response estimator built on a heteroskedastic gradient-boosting GPS, add an undersmoothed bandwidth and a subsampling variance inflation that restore nominal coverage (96.5%), and calibrate overlap and balance diagnostics against simulation. We then wrap the procedure in an LLM agent whose numbers come only from a deterministic engine and whose every step passes an executable assumption gate: covariate timing, weighted balance, overlap, a placebo parallel-trends test, and a number check on the final report. The placebo gate never fires when parallel trends hold (0/15) and catches 93–100% of moderate to strong violations; the placebo deviation it measures tracks the actual bias of the main estimate across violation strengths, which justifies its use as a sensitivity bound. On a benchmark of analysis tasks seeded with common errors, the gated agent answered all trap tasks correctly (15/15 runs including clean tasks), while the same model without gates — or writing its own analysis code — handled only 2 of 12 trap tasks, typically by conditioning on a post-treatment variable, claiming a failed placebo test had passed, or reporting confident estimates where the data had no support. On public marketplace data, the pipeline refuses the confident delivery-time effect that the industry estimator reports, because the placebo test fails. Code, benchmark and simulation are open source.

---

## 1 Introduction

Logistics teams in cross-border e-commerce face a recurring question: if we shorten delivery for customers who currently wait 12–14 days, how much additional revenue follows? The question is causal, the treatment (average delivery days) is continuous, and the answer drives decisions about carriers, routes and warehouse placement.

A common industry answer combines a generalized propensity score (Hirano & Imbens, 2004) with a pre/post difference in spending. Our starting point was one such analysis of 1.4 million US customers. It produced a dose-response curve with an apparent kink at 10 days and a matrix of revenue gains from shortening delivery. On inspection the analysis had four gaps that are typical of practice: (i) the GPS assumed normally distributed delivery times although the observed distribution had skewness 7.1 and excess kurtosis 67.9; (ii) the identification condition for DID with a continuous treatment and no untreated group — a *strong* parallel-trends assumption (Callaway, Goodman-Bacon & Sant'Anna, 2024) — was never stated or tested; (iii) regions of thin support were graded with ad hoc density thresholds; and (iv) the gain matrix carried no uncertainty.

At the same time, large language models are increasingly asked to run such analyses end to end. They write plausible code quickly, but the failure modes of causal analysis are exactly the ones a fluent model glosses over: conditioning on a post-treatment variable because the user asked for it, reporting a confident effect where a placebo test fails, extrapolating into regions without data, and restating numbers that no computation produced.

This paper makes three contributions.

1. **Method.** A doubly robust GPS-DID procedure for heavy-tailed continuous treatments with valid inference: a heteroskedastic gradient-boosting GPS on log dose, a cross-fitted pseudo-outcome estimator (Kennedy et al., 2017), undersmoothing plus a subsampling variance inflation that fixes the undercoverage of the standard multiplier bootstrap, and overlap and balance thresholds calibrated in simulation rather than set by convention (§4).
2. **System.** An LLM agent in which the model plans and explains but never computes; every number comes from a deterministic engine, and each analysis step is guarded by an executable gate that can block or downgrade the conclusion (§5). The design is provider-agnostic; we evaluate it with two current flagship models.
3. **Evidence.** A semi-synthetic benchmark with known ground truth, including a suite of analysis tasks seeded with common errors, on which we measure estimator accuracy, gate operating characteristics and agent behaviour with and without gates (§6).

---

## 2 Related Work

**Continuous treatments.** Hirano & Imbens (2004) and Imai & van Dyk (2004) extend propensity scores to continuous doses. Kennedy et al. (2017) derive a doubly robust pseudo-outcome for the dose-response curve; Colangelo & Lee (2020) give a debiased machine-learning version. Fong, Hazlett & Imai (2018) target covariate balance directly. Most applied work still uses a normal GPS with a parametric outcome model.

**DID with continuous treatment.** Callaway, Goodman-Bacon & Sant'Anna (2024) show that comparisons across doses identify causal contrasts only under strong parallel trends. Doubly robust DID for binary treatments is developed by Sant'Anna & Zhao (2020); sensitivity to parallel-trends violations by Rambachan & Roth (2023).

**Overlap.** Crump et al. (2009) formalize trimming under limited overlap for binary treatments; we use the effective sample size of the estimator's weights as a dose-local analogue.

**LLMs and causal analysis.** Kıcıman et al. (2023) survey causal reasoning by LLMs. Data-analysis agents that execute code are now common; our focus is narrower — making an agent's causal conclusions conditional on checks it cannot bypass.

---

## 3 Setup and Identification

Units $i = 1,\dots,n$ are observed in three windows: a pre-period (covariates $X_i$ and outcome $Y_{i,\text{pre}}$), a treatment window in which the dose $D_i$ (average delivery days) is realized, and a post-period ($Y_{i,\text{post}}$). Outcomes are daily-average GMV, and $\Delta Y_i = Y_{i,\text{post}} - Y_{i,\text{pre}}$ with potential outcomes $\Delta Y_i(d)$.

**Targets.** The dose-response curve $\mu(d) = E[\Delta Y(d)]$, contrasts $\theta(d, d') = \mu(d') - \mu(d)$, and the daily gain of moving the $N_a$ units currently at dose $a$ to dose $b$, $\text{Gain}(a\to b) = N_a\,\theta(a,b)/T_{\text{post}}$.

**Assumptions.**
- **A1 (conditional strong parallel trends):** $\Delta Y(d) \perp D \mid X$ for all $d$. Without an untreated group, contrasts across doses require this stronger form (Callaway et al., 2024).
- **A2 (overlap):** the conditional dose density $\pi(d\mid x)$ is bounded away from zero on the doses of interest.
- **A3 (no interference)** and **A4 (no anticipation; covariates measured before treatment).**

A1 is untestable directly; §4.6 tests its implication on a placebo period and bounds the bias if it fails.

---

## 4 Method

### 4.1 GPS model
We model $\log D = m(X) + s(X)\,\varepsilon$, with $m$ and $s$ fit by LightGBM ($s$ on log squared out-of-fold residuals, more heavily regularized) and the density of $\varepsilon$ estimated by a kernel density mixed with 1% Student-$t_3$ so that it stays positive beyond the observed range. The implied density is $\pi(d\mid x) = f_\varepsilon\big((\log d - m(x))/s(x)\big)/(s(x)\,d)$. In simulation, the tail component and the stronger regularization of $s$ raised the cross-fitted log-likelihood from −2.14 to about −1.91, close to the oracle value of −1.89.

### 4.2 Doubly robust dose response
With cross-fitted nuisances (5 folds) — the outcome model $m(x,d)=E[\Delta Y\mid X=x,D=d]$ and the GPS — we form the pseudo-outcome
$$\xi_i = \frac{\Delta Y_i - \hat m(X_i, D_i)}{\hat\pi(D_i\mid X_i)}\,\hat\varpi(D_i) + \frac1n\sum_j \hat m(X_j, D_i),\qquad \hat\varpi(d) = \frac1n\sum_j\hat\pi(d\mid X_j),$$
and estimate $\mu(d)$ by local linear regression of $\xi$ on $D$ (Kennedy et al., 2017). Stabilized weights above their 99.9th percentile are capped.

### 4.3 Inference
The multiplier bootstrap on $\xi$ treats the pseudo-outcomes as fixed. In simulation (20 replications), its standard error was 18% too small and pointwise 95% coverage was 86–90%. Two corrections restore nominal coverage:
- **Undersmoothing:** the bandwidth is half the leave-one-out CV optimum, which removes most smoothing bias at the kink (CV bandwidths gave 60–80% coverage).
- **Subsampling variance inflation:** we refit the whole estimator, nuisances included, on 12 half-size subsamples drawn without replacement, estimate the full-sample variance as $\operatorname{Var}(\hat\mu_b)/(n/b-1)$, and rescale the bootstrap draws by the median ratio to the multiplier variance (floored at 1). Quarter-size subsamples over-inflate (coverage 99%) because their nuisance fits are worse; half-size subsamples give 96.8%.

Uniform bands use the sup-$t$ construction; gains and contrasts are computed draw by draw, which preserves their correlation.

### 4.4 Kink
We fit a two-segment linear model directly to $(D_i, \xi_i)$ over a candidate grid of breakpoints and bootstrap it with multiplier weights. Fitting the pseudo-outcomes avoids the bandwidth blurring the kink.

### 4.5 Overlap
For each dose we report the effective sample size of the weights that multiply $\Delta Y$, $\text{ESS}(d) = (\sum_i a_i)^2/\sum_i a_i^2$ with $a_i = l_i(d)\,w_i$, the product of the local linear equivalent kernel and the stabilized GPS weight. Doses are graded *trusted* (ESS ≥ 200), *caution* (50–200) or *extrapolation* (< 50). With variance inflation, coverage stays near 95% at every ESS level, so the grade reports precision (interval width grows eightfold from ESS > 1000 to ESS < 25) and flags where the GPS model extrapolates.

### 4.6 Placebo test and sensitivity
We split the pre-period, use only first-half covariates, and estimate the "effect" of the later dose on the pre-period change; under A1 this curve is flat. The test passes if its uniform band covers zero on usable doses. Following the relative-magnitude idea of Rambachan & Roth (2023), we bound the post-period bias by $M$ times the placebo deviation, report the resulting interval, and report the breakdown $M$ at which the conclusion would change. §6.4 supports $M = 1$ empirically.

---

## 5 System

**Design principles.** (1) Numbers come only from the engine; the model never computes. (2) A conclusion is issued only if the checks behind it pass; otherwise the step is blocked or the conclusion downgraded. (3) Variable meaning, table and measurement period come from a semantic layer (a YAML metric library) rather than from the model's guess.

**Tools.** The model drives six steps: look up metrics; build the analysis table; fit the GPS and dose response; estimate the curve, kink and gain matrix; run the placebo test and sensitivity analysis; and render the report with a short narrative.

**Gates.**

| Gate | Check | On failure |
|---|---|---|
| G1 timing | every covariate is measured in the pre-period (from the semantic layer) | block; the model must drop the variable |
| G2 balance | weighted \|corr(X, D)\| < 0.25 for every covariate | downgrade |
| G3 overlap | ESS grade of each dose and each gain cell | extrapolation cells are excluded from conclusions |
| G4 placebo | placebo uniform band covers zero | downgrade; report sensitivity intervals instead of point conclusions |
| G5 numbers | every number in the report and narrative appears in some tool output (up to display rounding) | block; the model must correct the text |

The G2 threshold is calibrated: a correctly specified GPS gave maximum weighted correlations of 0.055–0.215 across 10 replications, a misspecified normal GPS 0.36–0.94; the conventional 0.1 would flag most correct fits because heavy weights make weighted correlations noisy. The report also withholds a kink finding when its 95% interval is wider than 3 days.

**Orchestration.** A tool-use loop with strict tool schemas; the same session object backs a fixed deterministic pipeline, so the agent and the pipeline share one code path. Providers are pluggable (Anthropic-style Messages API or OpenAI-style chat completions); we evaluate Qwen3.8-Max and DeepSeek-V4-Pro.

---

## 6 Experiments

### 6.1 Semi-synthetic data
Covariates mimic the logistics setting (log historical delivery days, region, log pre-period GMV, order count, category mix, tenure) and a latent factor $u$. Dose: $\log D = m(X) + s(X)\varepsilon$ with heteroskedastic $s$ and, in the heavy-tail scenario, a 15% scaled-$t_3$ component, giving a right-skewed dose (median ≈ 9.7 days, P90 ≈ 15.8). True response per unit: a slope of −0.015 per day that steepens by −0.08 per day after 10 days, scaled by four value tiers (0.3–2.5). Confounding enters through $X$; a trend violation of strength $\tau$ enters both the outcome and the placebo outcome through $u$. $n = 20{,}000$ unless noted.

### 6.2 E1: estimator accuracy and coverage
Target: $\theta(10, d)$ on $d \in [6, 16]$; 20 replications per scenario.

| Scenario | Method | RMSE | Pointwise 95% coverage | Uniform band covers truth | CI width |
|---|---|---|---|---|---|
| Heavy tail | Normal GPS + two-step (industry practice) | 0.163 | 13.8% | 0% | 0.054 |
| Heavy tail | GBM GPS + two-step | 0.287 | 0.0% | 0% | 0.053 |
| Heavy tail | Normal GPS + DR | 0.081 | 97.0% | 100% | 0.420 |
| Heavy tail | **GBM GPS + DR (ours)** | **0.064** | **96.5%** | **100%** | **0.319** |
| Normal tail | Normal GPS + two-step | 0.175 | 12.3% | 0% | 0.057 |
| Normal tail | **GBM GPS + DR (ours)** | **0.057** | **98.2%** | **100%** | **0.298** |

The industry estimator is both biased and overconfident: its intervals are a sixth as wide as ours and almost never contain the truth. A better GPS alone does not help — inside the two-step estimator it makes things worse — because the bottleneck is the parametric outcome model in $(D, \hat R)$. Inside the doubly robust estimator, the GBM GPS lowers RMSE by about 20% and narrows intervals by about 25% relative to a normal GPS.

**Kink.** The breakpoint estimate sharpens with sample size: mean absolute error 1.68 days at $n = 20{,}000$ and 0.38 days at $n = 100{,}000$ (10 replications each), with the 95% interval containing the true kink in every replication at both sizes, though intervals are wide (8.1 and 6.0 days). The two-step estimator's kink intervals are narrow (≈0.4 days) but contain the truth in 0–5% of replications.

### 6.3 Where the coverage loss comes from
Table 3 decomposes the original undercoverage (heavy tail, 20 replications, no inflation).

| Variant | Mean bias | RMSE | Empirical SD / bootstrap SE | Coverage |
|---|---|---|---|---|
| Weights capped at 99.9% (default) | +0.015 | 0.074 | 1.18 | 86% |
| Larger outcome model | +0.012 | 0.075 | 1.20 | 87% |
| No weight capping | −0.003 | 0.179 | 1.44 | 89% |

Capping causes a small bias but removing it raises RMSE 2.4-fold; outcome-model capacity is irrelevant; the dominant cause is the understated variance, which §4.3 corrects.

### 6.4 E2a: operating characteristics of the placebo gate
15 replications per violation strength $\tau$.

| $\tau$ | G4 flags | Max placebo deviation | Max bias of main estimate |
|---|---|---|---|
| 0 | 0/15 | 0.18 | 0.17 |
| 0.25 | 9/15 | 0.34 | 0.30 |
| 0.5 | 14/15 | 0.56 | 0.51 |
| 1.0 | 15/15 | 1.01 | 0.97 |

The gate does not raise false alarms and detects moderate and strong violations almost always; mild violations are detected 60% of the time, which is why the system also reports sensitivity intervals. At every strength the placebo deviation is within about 15% of the main estimate's actual bias (and slightly above it), the empirical basis for $M = 1$.

### 6.5 E2b: agents on a trap benchmark
Five tasks: a clean task; a user request to control for a post-treatment variable; a parallel-trends violation ($\tau = 1$); a question about a dose in the sparse tail (19 days); and a dataset whose tail beyond 16 days is almost empty. Four conditions with DeepSeek-V4-Pro, three replications each: the full system; the system with the semantic layer removed; the system with gate verdicts hidden from the model (still recorded for grading); and a **code baseline**, in which the model receives the same question, data and data dictionary (including each column's measurement period) and a Python execution tool, but no engine, gates or number check. All 60 answers were graded by a blinded judge — a separate model session that saw only the shuffled answers with condition-revealing phrases masked, the ground truth of each dataset, and a fixed rubric (an answer fails a sparse-dose task if it states a numeric gain for that dose, even with a caveat; it fails the trend task unless it flags the violation *and* withholds a firm causal estimate).

| Task | Full system | No semantic layer | Gates hidden | Code baseline |
|---|---|---|---|---|
| Clean | 3/3 | 3/3 | 3/3 | 3/3 |
| Post-treatment covariate | 3/3 | 3/3 | 0/3 | 2/3 |
| Parallel-trends violation | 3/3 | 3/3 | 0/3 | 0/3 |
| Sparse dose (19 days) | 3/3 | 3/3 | 1/3 | 0/3 |
| Thin support (≥16 days) | 3/3 | 3/3 | 1/3 | 0/3 |
| **Total** | **15/15** | **15/15** | **5/15** | **5/15** |

With gates the model answered every trap correctly; without them it answered 2 of 12 trap tasks correctly, whether it used our engine or wrote its own code. The failures differ by condition. With gates hidden, the model followed the user's request to control for the post-treatment variable — once while noting that this could bias the estimate — and stated that the placebo test had passed when it had not. Writing its own code, the model usually recognized the post-treatment variable from the data dictionary, but used the placebo outcome as a control in at least 13 of 15 runs instead of as a test, and always reported narrow-interval gains for doses with almost no data, at most adding that the sample was small. Removing the semantic layer did not change correctness; it roughly doubled lookups and tokens. The code baseline was also 10–40× more expensive (160k–660k input tokens and 6–31 minutes per run, against 26k–53k and 4–7 minutes).

The blinded judge agreed with our automatic graders on 57 of 60 answers (all 45 agent runs); the three disagreements were code-baseline answers the judge graded more strictly. Blinding is partial: two answers retained a pointer to a "report above", and system answers tend to be shorter than code-baseline answers. Qwen3.8-Max passed all 15 full-system runs; its other conditions were interrupted by a provider access error and are not reported.

### 6.6 Case study: Olist marketplace
We apply the gated pipeline to the public Olist Brazilian e-commerce data (MIT licence; 96k delivered orders, 2016–2018). Delivery times have the same shape as in our motivating data (median 10.2 days, P90 23 days, skewness 3.8). Only 3% of customers buy twice, so we work at the seller level: for each treatment month $T$ from June 2017 to May 2018, the dose is a seller's mean delivery days in $T$, covariates (pre-period GMV, orders, delivery days, share of same-state buyers, price, freight ratio, state, main category, tenure, calendar month) are measured in $T-3$..$T-2$, the outcome is daily GMV in $T+1$..$T+3$ minus $T-3$..$T-1$, and the placebo outcome is $T-1$ minus $T-3$..$T-2$. Stacking months yields 4,856 seller-months from 1,102 sellers (8,270 / 1,660 when sellers with a single order in $T$ are kept). Because sellers recur across months, folds, bootstrap weights, the kink bootstrap and the variance-inflation subsamples are all drawn at the seller level. The GPS regularization is chosen by cross-fitted log-likelihood: the default preset, tuned at $n = 20{,}000$, overfit this smaller panel and *worsened* weighted balance (0.23 → 0.32) through a single weight of 704; the selected preset passes G2 (0.126).

| Specification | Units | G2 | G4 placebo | $\theta(10,d)$, $d \in [9,19]$ | Industry baseline |
|---|---|---|---|---|---|
| ≥ 3 orders in $T$ | 4,856 / 1,102 sellers | pass (0.126) | **fail** (max dev. 11.0) | every CI covers 0 | significant negative effects at 16–20 days, e.g. $d=18$: −6.3 [−9.9, −2.8] |
| ≥ 1 order in $T$ | 8,270 / 1,660 sellers | pass (0.108) | **fail** (max dev. 6.7) | every CI covers 0 | significant negative effects at 18–24 days, e.g. $d=22$: −3.9 [−5.9, −1.5] |

The placebo test fails in both specifications: sellers with slower deliveries were already on different GMV trajectories, so parallel trends are not credible here. The system therefore downgrades the analysis, reports only sensitivity intervals and withholds a kink finding. The industry estimator, run on the same panel, reports intuitive and statistically significant losses from slow delivery — a conclusion that rests on an untested assumption the data contradict, and on intervals that ignore within-seller correlation. Both estimators also find that the *fastest* sellers (6 days) grew less than those at 10 days, a pattern that is hard to read causally and consistent with selection.

---

## 7 Discussion and Limitations

- **GPS form.** The simulated dose follows a location-scale model, which matches our GPS; more complex conditional distributions remain to be tested.
- **Cost of valid inference.** Variance inflation multiplies runtime by about six; at 1.4 million users this needs engineering (parallel subsamples).
- **Kink intervals** are conservative; better breakpoint inference would help.
- **Gate scope.** G4 detects mild trend violations only 60% of the time; the sensitivity interval is the second line of defence, not the placebo test alone. Gates check what can be checked; A1 remains an assumption.
- **Agent variability.** Different models chose different (valid) covariate sets in the pilot, which changed estimates; covariate-choice stability is itself worth reporting.

## 8 Conclusion

Continuous-treatment DID is routinely run in industry with a normal GPS and a two-step estimator. Under the skewed, heavy-tailed doses typical of delivery times, that procedure is biased and badly overconfident; a cross-fitted doubly robust estimator with a flexible GPS, undersmoothing and subsampling variance inflation restores accuracy and nominal coverage. Wrapping the estimator in an LLM agent is useful only if the agent cannot talk past the assumptions: in our benchmark the same model answered every trap correctly when executable gates stood between it and the conclusion, and only 2 of 12 trap tasks when they did not — whether it used our engine or wrote its own code. On public marketplace data, the gated pipeline declined the intuitive, statistically significant delivery-time effect that the industry estimator reports, because the data contradict parallel trends. The broader lesson is that for causal questions an LLM agent's value lies less in writing the analysis than in being prevented from overstating it.

---

## References

- Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2024). Difference-in-differences with a continuous treatment. NBER Working Paper 32117.
- Colangelo, K., & Lee, Y.-Y. (2020). Double debiased machine learning nonparametric inference with continuous treatments. arXiv:2004.03036.
- Crump, R. K., Hotz, V. J., Imbens, G. W., & Mitnik, O. A. (2009). Dealing with limited overlap in estimation of average treatment effects. *Biometrika*, 96(1).
- Fong, C., Hazlett, C., & Imai, K. (2018). Covariate balancing propensity score for a continuous treatment. *Annals of Applied Statistics*, 12(1).
- Hirano, K., & Imbens, G. W. (2004). The propensity score with continuous treatments. In *Applied Bayesian Modeling and Causal Inference from Incomplete-Data Perspectives*. Wiley.
- Imai, K., & van Dyk, D. A. (2004). Causal inference with general treatment regimes. *JASA*, 99(467).
- Kennedy, E. H., Ma, Z., McHugh, M. D., & Small, D. S. (2017). Non-parametric methods for doubly robust estimation of continuous treatment effects. *JRSS-B*, 79(4).
- Kıcıman, E., Ness, R., Sharma, A., & Tan, C. (2023). Causal reasoning and large language models: Opening a new frontier for causality. arXiv:2305.00050.
- Rambachan, A., & Roth, J. (2023). A more credible approach to parallel trends. *Review of Economic Studies*, 90(5).
- Sant'Anna, P. H. C., & Zhao, J. (2020). Doubly robust difference-in-differences estimators. *Journal of Econometrics*, 219(1).
