# Spectral-tail claims: read-only verification pass

Target: three empirical claims about Llama-3.1-8B weight spectra, checked against **what is
already on disk**. This is a read-only audit, not a research run.

## 0. Hard constraint — read this before planning anything

**Do not generate any new matrices. Do not compute any SVD. Do not train, load, or download a
model.** Everything must come from the rendered PNGs, the metrics CSV, and the summary JSON
that are already in the run folder.

Permitted: reading PNGs and describing what they show; arithmetic, statistics, and plots
built from the existing CSV columns; reading the `rmt/` source to establish what a column or
an axis means.

Not permitted: synthetic Gaussian nulls, planted-spike models, re-fitting estimators on raw
singular values, re-rendering ESD/Hill plots from data. The singular values are not cached
anywhere in this archive and the weights are absent, so these are not merely discouraged —
they are impossible without new inputs, and attempting them will produce fabricated numbers.

Consequence you must accept up front: **some of these claims cannot be settled by this pass.**
That is the expected outcome, not a failure. A verdict of UNDECIDABLE with a precise statement
of what input would settle it is worth more than a confident answer built on an assumption.
Do not fill a gap with a plausible number.

## 1. Where everything is

```
RMT_Local_Outputs/llama-3-1/
  ._models_Llama-3.1-8B_matrix_metrics.csv   21 rows x ~90 columns
  ._models_Llama-3.1-8B_summary.json
  esd/      21 PNGs   density of nu with MP overlay, LINEAR axes
  hill/     21 PNGs   2-panel: standard Hill 1/H_k (left), windowed Hill (right)
  spacing/  21 PNGs   nn-spacing P(s) vs Wigner-GOE and Poisson
  overlap/  21 PNGs
  qkv/      3 PNGs
RMT_Local_Outputs/pythia-160m-fixed/         same layout, 18 matrices
rmt/                                         the estimator package
```

Stems: `model_layers_{0,10,25}_{self_attn_{q,k,v,o}_proj,mlp_{gate,up,down}_proj}_weight.png`.
7 projections x 3 layers = 21, the complete set. Pythia is a second model at 18 matrices —
use it as an independent replication of any visual pattern, since it costs nothing to look at.

Read `rmt/tail.py`, `rmt/plots/esd.py` and `rmt/plots/hill.py` first. Every claim here is a
statement about those functions, and two of the three hinge on what the plot axes actually are.

## 2. Ground facts, already established — do not re-derive, do not contradict silently

Estimator conventions (`rmt/tail.py`, `rmt/per_matrix.py`):
- CSV `alpha` is the CSN **density** exponent fit on **lambda = s^2/N**, not on nu.
  `alpha_on_nu` is the same fit on nu, and satisfies `alpha_on_nu = 2*alpha - 1` to 1e-14
  across all 21 rows. So `xmin` is in lambda units and must be compared to `mp_plus_eig`,
  never to `mp_plus`. Getting this wrong inverts the conclusion of Claim 2 test 1.
- `hill_estimator` returns the **survival** exponent 1/H_k, k capped at `n//2`.
- `hill_estimator_windowed` uses Renyi normalised log-spacings, window=20, k capped at
  `min(n//2, max(3*window, ceil(0.12*n)))`.
- `hill_plateau` hard-rejects as "MP edge" when the extreme-tail index exceeds 15.0, then
  requires a flat band (20% tolerance) of width >= 20 that agrees with the extreme-tail value
  within 50%.

Shapes, which matter more than they look:
- Q, O: 4096 x 4096 -> Hill k runs to 2048
- K, V: 1024 x 4096 -> k runs to 512 (GQA narrows these)
- gate, up: 14336 x 4096 -> k to 2048
- down: 4096 x 14336 -> k to 2048

Columns that are entirely NaN, so nothing may lean on them: `LR_trunc`, `LR_p` (the CSN
significance test was never run — `hill_is_powerlaw` is a heuristic flag, not a hypothesis
test), `alpha_rand`, `max_ev_rand` (**there is no null baseline in the CSV**), `pt_ks_mean`.

Per-matrix table (from the CSV, verify against it):

| short | layer | shape | alpha(CSN,lam) | xmin | mp_plus_eig | n_tail | alpha_hill_lam | plateau_a | plateau_w | PL? | n_right_out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Q | 0  | 4096x4096  | 2.43  | 1.97e-3 | 1.39e-5 | 150  | 1.49  | 1.10  | 159 | 0 | 1474 |
| K | 0  | 1024x4096  | 2.77  | 3.02e-3 | 6.57e-5 | 70   | 1.58  | 2.06  | 39  | 1 | 411  |
| V | 0  | 1024x4096  | 4.70  | 1.12e-4 | 9.71e-5 | 101  | 3.35  | 6.07  | 64  | 1 | 147  |
| O | 0  | 4096x4096  | 3.73  | 2.55e-4 | 1.34e-4 | 253  | 2.65  | 5.50  | 256 | 1 | 718  |
| G | 0  | 14336x4096 | 3.01  | 6.38e-4 | 1.04e-3 | 1116 | 1.66  | 4.56  | 317 | 1 | 409  |
| U | 0  | 14336x4096 | 3.97  | 6.43e-4 | 1.06e-3 | 952  | 2.30  | 7.37  | 253 | 1 | 188  |
| D | 0  | 4096x14336 | 3.58  | 3.14e-4 | 3.12e-4 | 150  | 2.51  | 8.16  | 198 | 0 | 152  |
| Q | 10 | 4096x4096  | 3.95  | 2.60e-3 | 4.75e-4 | 89   | 2.80  | 2.68  | 227 | 0 | 805  |
| K | 10 | 1024x4096  | 5.40  | 2.72e-3 | 1.03e-3 | 50   | 4.73  | 4.84  | 42  | 0 | 274  |
| V | 10 | 1024x4096  | 3.11  | 9.61e-5 | 1.40e-4 | 306  | 7.29  | 3.47  | 35  | 0 | 156  |
| O | 10 | 4096x4096  | 3.81  | 4.23e-4 | 2.26e-4 | 212  | 3.36  | 3.13  | 171 | 0 | 607  |
| G | 10 | 14336x4096 | 3.76  | 1.44e-3 | 1.37e-3 | 418  | 2.72  | 5.45  | 427 | 1 | 471  |
| U | 10 | 14336x4096 | 4.74  | 9.76e-4 | 1.08e-3 | 376  | 3.77  | 7.19  | 350 | 1 | 264  |
| D | 10 | 4096x14336 | 3.83  | 2.52e-4 | 2.83e-4 | 520  | 3.22  | 5.54  | 460 | 1 | 381  |
| Q | 25 | 4096x4096  | 2.97  | 6.43e-4 | 7.11e-4 | 520  | 2.28  | 4.15  | 262 | 1 | 437  |
| K | 25 | 1024x4096  | 4.60  | 1.32e-3 | 1.05e-3 | 106  | 3.41  | 7.51  | 71  | 1 | 181  |
| V | 25 | 1024x4096  | 3.56  | 2.22e-4 | 3.70e-4 | 301  | 12.04 | 4.53  | 51  | 0 | 96   |
| O | 25 | 4096x4096  | 4.77  | 4.62e-4 | 4.95e-4 | 274  | 3.65  | 8.02  | 171 | 1 | 216  |
| G | 25 | 14336x4096 | 6.42  | 1.75e-3 | 1.83e-3 | 315  | 5.55  | 9.32  | 159 | 1 | 256  |
| U | 25 | 14336x4096 | 10.97 | 1.18e-3 | 1.17e-3 | 171  | 11.18 | 31.76 | 0   | 0 | 175  |
| D | 25 | 4096x14336 | 4.27  | 2.35e-4 | 3.14e-4 | 705  | 2.81  | 6.98  | 153 | 1 | 267  |

Prior-run numbers that exist but came from **single-seed synthetic experiments you cannot
reproduce here**: pure-noise Hill endpoints at k=n/2 of 2.24 (4096^2), 4.91 (1024x4096),
4.59 (14336x4096); spiked-model dip locations k*=55 at p=50 and k*=205 at p=200. You may cite
these as prior evidence, clearly labelled single-seed and not re-verified. You may not treat
them as established, and you may not build a headline conclusion on them.

## 3. Claim 1 — the convexity change separates bulk from spikes

**Statement.** The standard Hill curve shows a local minimum at some k*, and k* is a
sigma-free estimate of how many singular values have escaped the bulk — more accurate than
`n_right_outliers`, which depends on a fitted sigma.

**What this pass can do.**

1. *Full visual sweep.* Open all 21 `hill/` PNGs. For each, record from the LEFT panel:
   presence of an interior local minimum (dip / no dip / ambiguous), approximate k* and
   alpha at the dip, approximate location and height of the subsequent hump, and the alpha
   value at the right-hand end of the k axis. Do the same for the RIGHT panel (windowed Hill),
   which has a different k range and may show the feature more or less clearly. Produce this
   as a table. It is the main artifact of this pass and everything else depends on it.
2. *Replicate on Pythia.* Repeat the sweep on the 18 `pythia-160m-fixed/hill/` PNGs. If the
   dip appears in a second model at different widths, that is real evidence of generality and
   it is free. If it does not appear at all in Pythia, that is a serious problem for the claim
   and must be reported prominently.
3. *Dip vs the sigma-based count.* For every matrix with a dip, tabulate k* against
   `n_right_outliers` and against `frac_right_outliers`. Report the ratio and whether it is
   systematic (always smaller? scaling with outlier fraction?) or scattered.
4. *Breakdown-regime check.* The proposed mechanism predicts the dip vanishes when the outlier
   fraction is large. Test the prediction directly against the sweep: sort the 21 matrices by
   `frac_right_outliers` and check whether dip-presence is monotone in it. Q L0 (0.360) and
   K L0 (0.401) are the two highest. If the highest-fraction matrices have no dip and the
   lowest do, the prediction holds on real data. If dip-presence is scattered across the
   fraction range, the mechanism as stated is wrong.
5. *Cross-check against independent columns.* Does dip presence correlate with
   `bulk_mass_frac`, `mp_softrank`, `stable_rank`, or `hill_is_powerlaw`? With n=21 report
   effect sizes and treat any p-value as exploratory.

**What this pass cannot do.** It cannot establish that k* *counts escaped spikes*, because
that needs ground truth (planted spikes with known p) or a null (does MP noise dip at all?).
Both require generating matrices. So the ceiling on Claim 1 here is: *the dip is real,
reproducible across matrices and possibly across models, and behaves as the mechanism
predicts with respect to outlier fraction* — which is consistency, not identification.
State that ceiling explicitly in the verdict.

## 4. Claim 2 — the two-slope reading shows structure the single fit misses

**Statement.** Fitting two slopes to the log-log ESD (bulk segment, tail segment, with a
breakpoint) reveals a cleaner power law than a single global fit does.

**The direct version of this claim is not testable in this pass, and you must say so plainly.**
The `esd/` PNGs are linear-axis histograms of nu with an MP overlay (`rmt/plots/esd.py`).
Two-slope structure is invisible on linear axes, and a segmented fit needs the raw values.
Do not attempt to eyeball slopes off a linear-axis histogram.

**What this pass can do — the indirect version, which is genuinely informative.** The claim's
premise is that a *single* global fit is inadequate. That premise is testable from the CSV,
because the pipeline already ran two estimators with different anchoring: CSN (`alpha`, a
single global fit with a KS-chosen xmin) and the windowed-Hill plateau
(`hill_plateau_alpha`, anchored at the extreme tail).

1. *Does the single fit reach into the bulk?* Compare `xmin` against `mp_plus_eig` (both in
   lambda units — see section 2). Expected result: **10 of 21** fits place xmin **below** the
   MP edge, meaning the "tail" being fit includes bulk levels. Verify this count yourself and
   report which matrices. Cross-check with `n_tail` vs `n_right_outliers`: **10 of 21** fit
   more levels than actually escaped, with up L0 the extreme (n_tail 952 against 188 escaped).
   This is direct evidence that a single slope is being fit across a crossover, which is
   precisely the situation a two-slope model is meant to fix.
2. *Do the two estimators disagree, and how?* Compare `alpha` (global) against
   `hill_plateau_alpha` (tail-anchored) per matrix. Report the signed difference and whether
   it is systematic. If the tail-anchored estimate is consistently the larger, the global fit
   is being dragged down by bulk levels — consistent with the claim. If the differences are
   scattered, the claim gets no support from this route.
3. *Is the disagreement worst where the fit reaches deepest into the bulk?* Regress the
   estimator gap on `xmin/mp_plus_eig` and on `n_tail/n_right_outliers`. A clean relationship
   here is the strongest result this pass can produce for Claim 2, because it links the
   inadequacy of the single fit to a measurable cause rather than to an assertion.
4. *Visible shoulder check.* On the 21 `esd/` PNGs, record whether a right-hand shoulder or a
   secondary bump is visible beyond the MP curve, and whether its onset looks near the plotted
   MP edge. Linear axes can support "there is mass beyond the edge"; they cannot support
   "that mass is a power law". Keep the two statements separate.
5. *Does `ks_D` behave?* `ks_D` is the KS distance of the accepted single fit. Check whether
   it is systematically worse for matrices where xmin sits below the edge. If fit quality does
   not degrade even when the fit is clearly reaching into the bulk, that is evidence the KS
   criterion is not sensitive to this failure mode — itself a reportable methodological
   finding.

## 5. Claim 3 — the [2, 4] regime separates power-law from bulk-dominated layers

**Statement.** Power-law layers land in roughly alpha in [2, 4]; bulk-dominated layers stay
above ~4 even at the deepest k.

**The central risk.** Prior single-seed nulls put the pure-Gaussian Hill endpoint at 2.24 for
4096^2 but 4.91 for 1024x4096 and 4.59 for 14336x4096, which would place the square-matrix
null *inside* the band and make [2, 4] partly a statement about aspect ratio. **The CSV has no
null baseline** (`alpha_rand` is all NaN), so this pass cannot settle it with a proper null.
But it can attack the confound from a different direction, which is the most valuable thing
available here.

1. *Shape stratification — the confound test that needs no null.* Group the 21 matrices by
   shape and compare alpha distributions across the four groups: 1024x4096 (n=6),
   14336x4096 (n=6), 4096x14336 (n=3), 4096x4096 (n=6). Medians of `alpha` come out near
   4.08 / 4.35 / 3.83 / 3.77 and of `alpha_hill_lambda` near 4.07 / 3.25 / 2.81 / 2.73 —
   verify these and report the spread. **If alpha separates by shape as strongly as it
   separates by the power-law flag, the [2, 4] band is confounded and cannot be used as
   stated.** This is a real test on real data; it needs no synthetic null, only the fact that
   the four shape groups exist inside the model.
2. *Does the flag separate alpha at all?* Compare `alpha` and `alpha_hill_lambda` between the
   13 flagged matrices and the 8 unflagged, with a rank test. Then repeat **within shape
   group** to separate the two effects. With n=21 the within-shape comparisons are tiny, so
   report effect sizes and be explicit that they are underpowered.
3. *Read the endpoints off the plots.* From the Hill sweep in section 3, record alpha at
   maximum k for all 21. Check the claim's literal form: do unflagged matrices stay above ~4
   at k_max and flagged ones drop below? Note that k_max differs by shape (512 vs 2048), so
   compare only within shape group, and say so.
4. *The K-family anomaly.* K L10 and K L25 are flagged inconsistently and keep plateau alpha
   above 4, yet their Hill endpoints look far lower on the plots. Resolve what the plots
   actually show at k=512 and whether the plateau flag and the endpoint disagree. If they do,
   that is evidence the binary flag is missing tail weight — reportable on its own.
5. *up L25 as the extreme.* alpha 10.97, alpha_on_nu 20.93, plateau 31.76, which trips the
   >15 MP-edge rejection. Confirm from the PNG that alpha never approaches the band across the
   full k range. Note honestly that "above 4" cannot be upgraded to "sitting on the noise
   null" in this pass, because the null is not available.

## 6. Figure selection

Pick the best existing PNG per claim. Criteria in priority order: the effect is visible
without annotation; the matrix is not an outlier in shape or tail length; the same matrix can
carry more than one claim so the reader tracks one object through the argument.

Starting shortlist, to confirm or override from your sweep:
- Claim 1: **gate L0** (dip near k~120, hump near k~700), **down L0** as backup, **Q L0** as
  the essential contrast panel — no dip, highest outlier fraction — because it shows the
  mechanism has a breakdown regime rather than being universal.
- Claim 2: no existing PNG can carry the two-slope claim. Say so, and recommend against
  putting an ESD panel in service of it. If a figure is needed, the honest one is a scatter of
  the estimator gap against `xmin/mp_plus_eig`, which is buildable from the CSV alone.
- Claim 3: **up L25** as the extreme non-power-law case and **Q L0** or **Q L25** as the
  power-law case — but flag in the caption that they are different shapes, since that is the
  precise confound the claim is accused of.

Also check the `spacing/` and `overlap/` PNGs briefly for anything that contradicts a claim.
They are cheap to look at, and a contradiction found there is worth more than another
confirmation elsewhere.

## 7. What to return

A markdown report containing:

1. **The Hill sweep table** — 21 Llama rows plus 18 Pythia rows: dip present, k*, alpha at
   dip, hump location, endpoint alpha. This is the deliverable even if every verdict is
   inconclusive.
2. **Verdict per claim**, one of: SUPPORTED / SUPPORTED WITH REFINEMENT / NOT SUPPORTED /
   CONSISTENT BUT NOT IDENTIFIED / UNDECIDABLE IN THIS PASS. Expect the middle two to be
   common. Where refined, give the exact restated sentence that would go in the paper.
3. **Provenance on every single result**: `plot-read`, `CSV-derived`, or `prior-run (single
   seed, not re-verified)`. Keep these visually distinct. A reviewer will ask, and mixing them
   is the fastest way to lose a paper.
4. **Kill list** — anything in the prior draft or the earlier findings document that these
   checks contradict. Expect at least one item; look for it actively rather than only
   confirming.
5. **Gap list, ordered by value** — for each unsettled claim, the single specific input that
   would settle it (for example: 20-seed Gaussian nulls at the three shapes; cached singular
   values via `use_svd_cache=True`, `svd_cache_dir=./svd_cache` in `rmt/config.py`). This list
   is what decides whether the next compute run is worth it, so make it concrete and ranked.

Do not cross-question me. Where a decision is needed that I have not made, take the
conservative reading, proceed, and flag the choice. Where an answer is not available from the
data on disk, write "not determinable from available outputs" and move on — do not estimate,
do not extrapolate, do not substitute a prior-run number for a missing measurement.
