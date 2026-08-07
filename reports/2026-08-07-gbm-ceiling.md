# The closed form is not the ceiling — but all of its regret is interaction structure, and the GBM pays for it in noise robustness

**2026-08-07** · issue [#6](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/6) · benchmarks the [#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3) refit ([2026-08-07 report](2026-08-07-distance-refit.md)) · **benchmark only — explicitly not a production candidate**

| | |
|---|---|
| **0.54 m** | median lat/lng test error of the LightGBM benchmark on the published 79,029-row split — the shipped blend D answers **0.93 m**, so D sits **+74%** above the ceiling (the 2021 baseline +169%). The 10–15% "essentially free of regret" verdict does **not** hold |
| **0.930 ≈ 0.934 m** | a GBM restricted to the *single* vertical signal (`sv_image_y`) lands exactly on blend D: the closed form has essentially **no 1-D regret**. All the headroom is interaction/context structure |
| **< 5 mm** | what dropping any feature group except `sv_image_y` costs the full GBM — the headroom has **no single carrier**; it lives in a redundant pool led by the resolution/era axis |
| **4–5× / 1.8×** | the GBM's click-noise degradation vs blend D at σ = 2 px / 10 px: the structure that buys the ceiling is exactly the structure noise destroys first |

> Reproduce every number here in one command each:
>
> ```bash
> python python/run_gbm_ceiling.py --write      # ~6 minutes, offline, deterministic
> python python/gbm_ceiling_figures.py          # figure 19
> pytest tests/test_gbm_ceiling_findings.py     # the findings, locked
> ```

## §1 · Goal

Issue #6 asks one calibration question of the #3 refit: is the shipped closed form within
10–15% of what these inputs can support — "essentially free of regret" — or is there real
headroom that a flexible learner can see? This report answers it with a deliberately
unshippable benchmark: a LightGBM regressor given the same inputs, split, and scoring as
every #3 rung, so any gap it opens (or fails to open) is attributable to modeling capacity
alone. The GBM exists to calibrate ambition — to bound what any future refit could hope to
gain, and to price what chasing that gain would cost.

## §2 · Questions

The report and its code (`python/run_gbm_ceiling.py`) were set up to answer six questions,
each locked by the findings tests:

- **Q1 — The ceiling.** Is blend D within 10–15% of what these inputs support? → **§5**:
  no — D sits **+74%** above the GBM's 0.536 m test median.
- **Q2 — 1-D regret.** Does the closed form leave anything on the table *as a 1-D curve*?
  → **§5**: essentially nothing — a GBM given only `sv_image_y` lands on D
  (0.930 ≈ 0.934 m); all the headroom is interaction/context structure.
- **Q3 — The carriers.** Which features carry the headroom — is a nameable closed-form term
  hiding in it? → **§6**: no single carrier — every drop-one except `sv_image_y` costs
  < 5 mm; it is a redundant pool led by the resolution/era axis.
- **Q4 — Where it lives.** Where in true distance does the gap sit? → **§7**: everywhere,
  widening with distance — and the GBM holds the 10–15 m bin the blend's saturation trades
  away.
- **Q5 — The price.** Is the GBM's edge robust to click noise? → **§8**: no — it degrades
  4–5× faster than blend D at σ = 2 px; the ceiling is fragile exactly where production
  inputs are noisy.
- **Q6 — The recommendation.** Does any of this change #3's shipping recommendation?
  → **§9**: no — the closed form keeps interpretability, boundedness, and the better noise
  response; the GBM must not ship.

## §3 · Dataset and harness (identical to #3 by construction)

Every number is computed from the committed `data/labels-*-latlng.csv.gz` — the 2026-08-05
reconstruction of the 2017–2020 depth-derived label placements
([#1](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/1), reproduction-grade:
exact 395,147 cleaned rows) — cleaned by the exact 2021 pipeline and partitioned by the
published R split (`tests/fixtures/r-baseline/split_*.csv.gz`, 316,118 train / 79,029 test).
Ground truth is the stored depth-derived `lat`/`lng`; its caveats (curb-height bias, occlusion
clusters, item G, the float32 grid) are §6 of the [#3 report](2026-08-07-distance-refit.md) and
travel with everything below. No new data, no network.

The harness is `run_distance_refit.py`'s, reused by import, not copied:

- **The GBM predicts distance only** — that is what #6 bounds. It is paired with the identical
  heading half as every #3 rung (exact POV inversion + the one era constant, +0.7198°), and its
  distance is clipped to the same 50 m cap, so its lat/lng numbers sit directly in the #3
  matrix.
- **Scoring is identical**: turf-style spherical destination, error = haversine meters to the
  depth-truth position.
- **Comparability is asserted, not assumed**: the runner refits A (status quo) and blend D
  in-process and requires them to equal `data/distance-refit-summary.json` to float precision
  before anything is written; the findings tests re-check that from the committed artifacts.
  The click-noise sweep is not a mirror of #3's — it *is* #3's, called with the GBMs handed in
  as extra predictors, so the A/D deltas reproduce exactly by construction rather than by two
  implementations staying in step. (A findings test still checks the equality; it can now only
  fail for a real reason.)

## §4 · The benchmark

LightGBM 4.7 (`lightgbm` is in `requirements.txt` marked benchmark-only), features from the
issue: raw `sv_image_y`, height-normalized `sv_norm = sv_image_y · 6656/pano_height`,
`canvas_x/y`, `zoom`, `label_type` (native categorical), `heading`/`pitch`, `pano_height`.
**DC has no `pano_height`** (58–59% of both splits — the column postdates the DC schema), so
`sv_norm` and `pano_height` are NaN there; both are passed natively (LightGBM routes missing
values), no rows dropped, nothing imputed. Two objectives, mirroring the ladder's loss rider:
`regression_l1` (aligned with the published median metric) and `regression_l2`; L1 wins the
median (0.538 vs 0.596), same story as the ladder's L1 column.

Determinism and honesty: fixed seed 666 everywhere, `deterministic=true`,
`force_row_wise=true`, no bagging or feature subsampling; early stopping uses a seeded 90/10
carve of the **train** split only, then the booster is refit on the full train split for
exactly the stopped round count — the test split is never consulted during fitting.

One caveat stated plainly rather than buried: the "ceiling" quoted is the best of the three
main variants *on test*, so it is optimistic and D's 74% is an **upper** bound on its regret —
the anti-conservative direction for this report's own "the gap is large" conclusion. It does
not carry the result. `d_over_gbm_gap_pct_by_variant` publishes what the selection is worth:
D is +74.1% over `gbm_dep_l1`, +73.6% over `gbm_l1` (they are 1.6 mm apart), and still **+56.5%
over `gbm_l2`**, the variant that loses. The answer to #6 — "not within 10–15%" — is the same
under every one. Note also that `drop_heading_pitch` scores 0.5346 on test, better than any of
the three; that is exactly what selecting on test looks like, and why the ablation rows are
reported as ablations and never as the ceiling.

## §5 · The matrix, and the answer to the ceiling question

Test split n = 79,029; heading half identical everywhere; A and D reproduced from #3 exactly:

| model | rounds | lat/lng med (m) | p90 | dist med (m) | p90 |
|---|---:|---:|---:|---:|---:|
| A ols (status quo form) | – | 1.4438 | 5.155 | 1.3955 | 5.139 |
| **D blend per-type l1 (shipped, #3)** | – | **0.9335** | **4.476** | **0.8713** | **4.453** |
| GBM l2 | 621 | 0.5964 | 3.395 | 0.4893 | 3.371 |
| GBM l1 | 1376 | 0.5378 | 3.291 | 0.4198 | 3.263 |
| GBM l1 + exact depression | 1341 | **0.5362** | **3.286** | 0.4174 | 3.256 |
| GBM, `sv_image_y` only | 150 | 0.9303 | 3.775 | 0.8576 | 3.754 |
| GBM, canvas x/y only | 101 | 2.7865 | 8.423 | 2.7692 | 8.418 |
| GBM, exact depression only | 122 | 0.9855 | 4.389 | 0.9278 | 4.368 |

**The answer is no**: blend D does *not* sit within 10–15% of the GBM — it sits **74.1%**
above it (0.9335 vs 0.5362 m; A_ols +169%). There is real conditional structure the geometry
is not using, and the tail improves too (p90 4.48 → 3.29 m, −27%).

**But the regret is not one-dimensional.** The two single-signal rows are the structural
finding of this benchmark: a free GBM given only `sv_image_y` reaches 0.930 m — statistically
on top of blend D's 0.934 — and one given only the exact depression angle reaches 0.985 m. The
shipped closed form extracts essentially everything a 1-D vertical model can (consistent with
#3's isotonic rung landing within 5 cm of the cotangent family). Every meter of headroom is
*interaction* structure. And handing the GBM the #5 exact projection as an explicit feature is
worth ~2 mm (0.5378 → 0.5362): the raw inputs already contain it, so the projection itself is
not where the advantage comes from either.

(A curiosity with a meaning: `sv_image_y`-only beats exact-depression-only by 5.5 cm median
and 0.61 m p90, although in a fixed frame the two encode the same angle. `sv_image_y` is the
column the 2017–2020 client actually fed into the depth lookup that *produced* the ground
truth, so it sits on the truth's causal path and carries the era client's quantization
artifacts that the truth also saw. Some fraction of the GBM's edge is therefore
truth-pipeline structure rather than scene geometry — unmeasurable from inside this dataset,
and one more reason to read the ceiling as an upper bound.)

## §6 · The ablation: what carries the headroom

Drop-one from the full L1 model (Δ test dist median, m; positive = worse without it):

| dropped | Δ dist med | Δ lat/lng med |
|---|---:|---:|
| `sv_image_y` | **+0.101** | **+0.071** |
| `zoom` | +0.000 | −0.001 |
| `label_type` | −0.000 | −0.001 |
| `sv_norm` | −0.001 | −0.000 |
| `pano_height` | −0.002 | −0.001 |
| `canvas_x/y` | −0.004 | −0.002 |
| `heading`/`pitch` | −0.005 | −0.003 |

Only `sv_image_y` is individually load-bearing, and even it costs just 0.10 m — `sv_norm` plus
the click geometry reconstruct most of it. Everything else is interchangeable: several drops
*improve* the median slightly. So the ablation's answer to "which features carry the gap" is:
**no single one** — it is a redundant pool of context around the vertical signal. By split
gain the pool is led by the resolution/era axis (`sv_norm` 8.2% + `pano_height` 5.6% — the
same rig/era confound #3's `B_log` term detected), then `heading` (2.4%, plausibly a
city/street-grid proxy), with canvas position, `pitch`, `label_type`, and `zoom` under ~1%
each (`sv_image_y`: 80%). There is no closed-form candidate hiding in this table: nothing here
suggests a nameable term that would move blend D meaningfully — the headroom is diffuse,
high-order, and partly pano-context-shaped.

## §7 · Where the gap lives: error vs true distance

Median lat/lng error by true-distance bin (m):

| bin (m) | n | A ols | blend D | GBM l1 |
|---|---:|---:|---:|---:|
| 0–2 | 138 | 4.49 | 4.33 | 3.87 |
| 2–5 | 7,405 | 2.63 | 0.53 | 0.38 |
| 5–10 | 35,884 | 1.33 | 0.72 | 0.41 |
| 10–15 | 21,933 | 0.92 | 1.16 | 0.65 |
| 15–20 | 8,069 | 2.14 | 1.50 | 1.20 |
| 20–30 | 4,290 | 6.52 | 3.81 | 2.39 |
| 30–50 | 1,310 | 17.23 | 13.26 | 8.36 |

![Figure 19 — left: median test error by true-distance bin (log scale); the GBM beats the blend in every bin, and holds the 10–15 m bin where the blend's saturation puts it behind even the status quo. Right: the seeded click-noise sweep of §8 — the GBM's degradation curve rises 4–5× faster than the blend's at small σ.](../figures/fig19-gbm-ceiling.png)

The GBM wins everywhere, but the *relative* gap widens with distance: −29% at 2–5 m, −44% at
5–15 m, −20/-37% at 15–30 m, −37% at 30–50 m — and the far field is exactly where the depth
truth is weakest (#3 §6: item G's rotated depth columns, occlusion clusters, the terrain
model's far-field softness). The bins that dominate the pooled median (5–15 m, 73% of test)
are also where D's one systematic weakness shows: its 10–15 m row (1.16 m) is worse than A's
(0.92 m) — the blend's saturation trades that region away — while the GBM holds 0.65 m there.
That one bin is the largest identifiable share of the pooled gap, and it is conditional
structure (which panos, which contexts run long), not a better 1-D curve, that buys it.

## §8 · The noise sweep: the ceiling is fragile

Same perturbation design, same seeded draws as #3 (verified exact on the A/D rows; figure 19,
right panel): Gaussian
click noise on `canvas_x/y`, every click-dependent feature re-derived (`sv_image_y` via the
fixed-frame px/deg scale, `sv_norm` and depression downstream), heading half unperturbed.
Δ median lat/lng error vs unperturbed (m):

| model | σ = 2 px | 5 px | 10 px |
|---|---:|---:|---:|
| A ols | 0.007 | 0.041 | 0.122 |
| blend D | 0.006 | 0.040 | 0.145 |
| GBM l1 | 0.028 | 0.097 | 0.259 |
| GBM l1 + depression | 0.030 | 0.098 | 0.258 |

Measured, not assumed — and the direction is clear: **the GBM is the more noise-sensitive
model**, 4–5× blend D's degradation at 2 px and ~1.8× at 10 px. Axis-aligned splits did not
buy robustness here; the fine conditional structure that creates the ceiling is the first
thing click noise destroys. (The ceiling conclusion itself survives: even fully degraded at
10 px the GBM's median, 0.796 m, is still below D's unperturbed 0.934 m.) For production —
where the input is a single human click, not the exact pixel the truth was computed from —
the closed form's flatter noise response is worth real accuracy back.

## §9 · What this changes, and what it does not

- **The #6 question is answered: there is a large gap** (D +74% over the ceiling), so the #3
  closed form is *not* within modeling-regret noise of what these inputs support. But the gap
  has no closed-form shape on offer: it is not the 1-D geometry (zero regret there), not the
  exact projection (+2 mm), not any single feature (drop-one < 5 mm) — it is diffuse
  interaction structure, concentrated beyond 10 m, partly rig/era-confounded, plausibly
  partly truth-pipeline artifact (§5's causal-path note), and twice as fragile under click
  noise.
- **The #3 recommendation stands.** Blend D keeps its virtues — 8 physical parameters, a JS
  one-liner, bounded by construction, the better noise response — and the benchmark shows the
  only thing it leaves behind is structure that would cost interpretability, robustness, and
  a model runtime to chase. If that trade is ever wanted, the 10–15 m band (D's one bin
  behind the status quo) is where to look first.
- **This model must not ship**: no JS runtime, no interpretable coefficients, and its edge
  partially rides the truth pipeline. It exists to calibrate ambition for Stage 3 and beyond,
  nothing else.

## Reproducing this report

```bash
pip install -r python/requirements.txt        # includes lightgbm (benchmark-only)
python python/run_gbm_ceiling.py --write      # ~6 min, offline, deterministic
python python/gbm_ceiling_figures.py          # figure 19 (renders the committed summary)
pytest tests/test_gbm_ceiling_findings.py     # the findings, locked
```

No network anywhere: committed CSVs, the R-fixture split, and the committed #3 summary (for
the comparability assertions) are the only inputs.

---

*Report generated with [Claude Code](https://claude.com/claude-code) (claude-fable-5); every
headline number is asserted by `tests/test_gbm_ceiling_findings.py` against
`data/gbm-ceiling-summary.json`, which regenerates deterministically from the committed data.*
