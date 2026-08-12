# The #6 ceiling was a scale, not scene structure — it does not survive a change of truth frame

**2026-08-10** · issue [#6](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/6),
second pass · re-examines the [2026-08-07 ceiling benchmark](2026-08-07-gbm-ceiling.md)
against the truth built in the [Stage 4 close-out](2026-08-07-modern-truth.md) ·
**benchmark only — still explicitly not a production candidate**

| | |
|---|---|
| **0.410 vs 0.498 m** | the shipped two-parameter closed form against the #6 booster plus one modern scale, on modern measured-plane truth, held-out panorama half. The ceiling does not merely shrink — it **inverts**, and the closed form also beats every richer recalibration tried and a booster trained on modern truth itself. Every paired cluster-bootstrap interval excludes zero |
| **+108% → +118%** | the booster's era-frame margin over the era blend, restated on modern truth. It does **not** break out of sample — which is what makes the row above a finding about the truth frame rather than about the model |
| **0.44 m → −0.01 m** | what the eight extra inputs buy the booster over the same booster given only `sv_image_y`: a large margin inside the era truth, nothing outside it. #6's headroom was interaction structure; this is that structure's value in another frame |
| **2.80 / 2.79 / 2.35 m** | the era truth's implied camera height where `pano_height` is absent (DC), at 6656-px panoramas, and at 8192-px ones. Not one scale — and the 8192 figure sits within 1 cm of both the modern measurement and the shipped constant. That heterogeneity is what the booster was reading |

> Reproduce every number here in one command each. Offline from a fresh checkout — the
> committed CSVs, the R-fixture split, and three committed summaries (`distance-refit`,
> `gbm-ceiling`, `modern-truth`) are the only inputs:
>
> ```bash
> python python/run_gbm_transfer.py --write     # 2-7 min, offline, byte-identical
> python python/gbm_transfer_figures.py         # figure 28
> pytest tests/test_gbm_transfer_contract.py    # the frame mapping and the harness
> pytest tests/test_gbm_transfer_findings.py    # the findings, locked
> ```

## §1 · Goal

#6 asked whether the #3 closed form sits within 10–15% of what its inputs support. The
[2026-08-07 benchmark](2026-08-07-gbm-ceiling.md) answered no: a LightGBM on the same
inputs and the same published split reached 0.536 m median lat/lng error against blend D's
0.934 m, a 74% gap, diffuse across features and concentrated beyond 10 m. That report also
flagged, in its §5, the thing it could not measure:

> "`sv_image_y` is the column the 2017–2020 client actually fed into the depth lookup that
> *produced* the ground truth … Some fraction of the GBM's edge is therefore
> truth-pipeline structure rather than scene geometry — **unmeasurable from inside this
> dataset**."

Three things have changed since, and together they make it measurable and make it worth
measuring:

- **A precedent.** Stage 4 found that era-fitted structure does not automatically survive
  a change of truth frame. The era fit's per-type height table — worth 4 cm on the era
  test split — bought *nothing* on modern truth, and the shipped coefficients dropped it
  for a single flat height. The GBM is the same species of claim at a hundred times the
  scale.
- **A second truth frame.** `data/modern-truth-labels.csv.gz` carries every feature the
  booster eats on post-2021 rows whose truth is a freshly sampled modern GSV ground plane.
  Those rows are disjoint from the era training split by the 2021-01-01 cutoff, so the
  #6 booster is fully held out on them.
- **A moved reference.** #6 benchmarked against "blend D (shipped)". Hours after it
  merged, Stage 4 replaced that with the flat two-parameter
  `final_coefficients` (height 2.3412 m, blend 11.25°). The report's comparison anchor is
  no longer what ships, and on era-frame metrics the shipped calibration is ~4 cm worse by
  construction — so #6's +74% *understates* the gap for the estimator that actually runs.

So the question this report answers is not "what is the ceiling" — that was answered — but
**is the ceiling real, or is it the era truth frame's own structure?**

## §2 · Questions

Each is locked by `tests/test_gbm_transfer_findings.py`:

- **Q1 — Does the booster simply break out of sample?** → **§5**: no. Against the era
  blend on modern rows it keeps its entire era margin (+108% → +118%), and it arrives
  nearly unbiased (signed −0.15 m) where the era blend arrives 1.07 m long.
- **Q2 — With one modern parameter each, who wins?** → **§6**: the closed form.
  0.410 m against 0.498 m, and against every richer recalibration.
- **Q3 — Is that just because the booster was given only one parameter?** → **§6**: no.
  An affine fit, a monotone quantile map, and a booster *trained on modern truth* all lose
  too; every paired bootstrap interval excludes zero.
- **Q4 — What was the ceiling made of, then?** → **§7**: a resolution-conditioned scale.
  The era truth implies 2.80 m of camera at DC and 6656 px but 2.35 m at 8192 px, and a
  booster given `pano_height` can answer on each subpopulation's own scale.
- **Q5 — Does anything transfer?** → **§8**: the tail, cleanly — p90 3.55 m → 2.80 m, and
  → 1.99 m for the modern-trained booster. The far field is murkier than #6's reading
  suggests: beyond 15 m the booster beats the *shipped* form, but so does the era blend
  simply by being biased 1.07 m long, so most of that deficit is the shipped saturation
  rather than conditional structure.
- **Q6 — Does this change what ships?** → **§9**: no, and it strengthens the Stage 4
  decision. It does change how much ambition a future refit should carry.

## §3 · Dataset and harness

**The boosters are #6's**, not re-specified ones. `run_gbm_transfer.py` refits them with
`run_gbm_ceiling.fit_gbm` — the same code, seed 666, the same two-pass early stopping — and
then *requires* them to reproduce the committed `data/gbm-ceiling-summary.json` era-test
medians to float precision before a single modern row is scored. If that assertion ever
fails, this report is not benchmarking the model #6 describes, and the run stops.

**The rows are Stage 4's**, not a new selection: the 2,655 gate-passing, human-placed rows
of `data/modern-truth-labels.csv.gz` (922 panoramas, 36 cities, all post-2021), truth =
`truth_m`, the horizontal ground range to a freshly sampled modern GSV plane. Its gates,
its circularity guard and its caveats are §§3–5 of the [Stage 4
report](2026-08-07-modern-truth.md) and travel with everything below.

**Nothing is refitted on modern data.** The only parameter this report ever fits is one
global scale per model, derived on a train half of panoramas and scored on the disjoint
half — the same split, seed and depression floor `modern_truth.remedy_check` used for the
blend, so the closed-form rows of §6's table are *asserted equal* to the committed Stage 4
remedy block rather than re-derived near it.

Two population facts that matter and are not hidden:

- **The rows are inside the era training support.** A booster cannot extrapolate — outside
  the training range every split is spent and the prediction flattens to an edge leaf — so
  a transfer test on out-of-support rows would measure the leaves, not the model. Under
  1.3% of rows fall outside the era range on *any* feature, and 8192-px panoramas, which
  dominate here (93%), are the era data's largest resolution group by far — 121,177
  cleaned rows at depression ≥ 5°.
- **Two label types are new.** `Crosswalk` and `Signal` postdate the era categorical, so
  433 rows (16.3%) reach the booster as a missing category. The shipped flat model carries
  no such handicap — it dropped `label_type` from the distance path entirely. §8 cuts the
  comparison on exactly that boundary so it cannot serve as an excuse. The one arm where
  this costs more than it looks is the modern-trained control of §6: it is fitted through
  the same era feature builder, so those two types are missing from its *training* data as
  well, not just at prediction time.

### The frame mapping, and why it is not a fudge

This is the one place the test could go quietly wrong. The era client stored `sv_image_y`
as a **fixed-frame offset from the horizon** — negative downward, in the 13312×6656 frame
the deployed coefficients were calibrated in — while the modern schema stores `pano_y`, an
absolute row index in the panorama's **real** raster. Both encode the same angle, so they
convert exactly:

```
sv_image_y  =  (pano_height/2 − pano_y) · 6656/pano_height  =  −depression_deg · 6656/180
```

Feeding raw `pano_y` instead would be #4765's defect reintroduced as a silent 23% error on
8192-px panoramas, with no exception anywhere. Three checks, all in the contract tests:

| check | result |
|---|---|
| algebraic, on the modern rows | mapped column equals −depression × 36.978 px/deg to **2 × 10⁻¹³ px** |
| against the #5 exact projection (canvas + POV, never touches `pano_y`) | median **+0.002 px**, p10/p90 ±0.33 px — under 0.01° |
| against 162,846 era rows carrying a real-pixel row, per height group | mapped residual **+15.0 px at 6656 and +14.6 px at 8192** — the same pano re-registration drift in both groups, against **+140 px** for the unmapped offset at 8192 |

Two height groups landing on the *same* small residual is what makes this the right
conversion rather than a fitted one.

The first two rows are computed by the run itself and stored in the summary's
`frame_mapping` block. The third is the only claim in this report that needs the whole era
frame rather than the scored rows, so it is locked by
`test_era_frame_residuals_match_the_published_table` — the row counts and both residuals to
0.005 px — which loads and cleans that frame and therefore runs under `RUN_SLOW=1`. The
default suite runs the same comparison on one committed city as an early warning.

## §4 · The three comparisons, kept apart

Every era-trained model inherits the era truth's scale, so a naive "score them all on
modern truth" would rank models by how badly that scale hurts them and call it a ceiling
test. The comparisons are therefore separated:

1. **Raw** (§5) — booster against the *era* blend on modern rows. Both carry the era
   scale, so the handicap is held constant and only the era margin is being asked about.
2. **Calibrated** (§6) — each side gets one modern parameter from a disjoint half. This is
   the question that matters.
3. **Generous** (§6) — the booster additionally gets an affine fit, a monotone quantile
   map, and a modern-trained sibling. This is where "you were unfair to it" gets answered.

## §5 · Raw transfer: the booster does not break

Pooled over all 2,655 gated human rows; no parameter fitted on either side (the shipped
flat model is shown for orientation only — its height was fitted on these rows, which is
why the honest comparison waits for §6):

| model | median \|err\| (m) | signed median | p90 | range slope |
|---|---:|---:|---:|---:|
| A deployed (2021 production) | 1.228 | −0.472 | 5.240 | −0.438 |
| D era blend (#3 Stage 2) | 1.291 | +1.067 | 3.976 | −0.341 |
| **GBM L1 (#6)** | **0.594** | −0.154 | 3.217 | −0.326 |
| GBM L1 + exact depression | 0.564 | −0.154 | 3.255 | −0.327 |
| GBM L2 | 0.709 | −0.320 | 3.973 | −0.340 |
| GBM, `sv_image_y` only | 1.080 | +0.992 | 3.289 | −0.247 |
| *D flat (shipped; in-sample here)* | *0.444* | *−0.172* | *4.127* | *−0.436* |

The era-frame margin survives intact: **+108% in the era frame, +118% here**. Whatever
this booster learned, it was not noise that evaporates the moment the rows change.

But one row gives the game away already. The full booster arrives at modern truth almost
**unbiased** (−0.154 m) while the era blend arrives **1.07 m long** — and the booster
restricted to `sv_image_y` arrives at **+0.99 m**, right beside the blend. The only
difference between those two boosters is the eight extra inputs, of which `pano_height`
and `sv_norm` are the resolution axis. The full booster is not predicting distance better;
it is predicting *which scale this row's truth was measured on*. §7 shows it was right to.

## §6 · Calibrated transfer: the ceiling inverts

Held-out panorama half (n = 1,362); every fitted parameter on either side comes from the
other half. The first four rows are `modern_truth.remedy_check`'s, asserted equal to the
committed Stage 4 table:

| model | parameters fitted on the train half | median \|err\| (m) | 95% CI | Δ vs D flat, paired | p90 |
|---|---|---:|---|---|---:|
| A deployed | none | 1.170 | – | – | 4.858 |
| D era blend | none | 1.279 | – | – | 3.603 |
| **D flat (shipped form)** | 1 (the camera height) | **0.410** | 0.376–0.456 | — | 3.553 |
| D rescaled | 1 (a factor on the height table) | 0.442 | 0.396–0.496 | −0.001 … +0.060 | 3.558 |
| GBM trained on modern truth | the whole booster, on 1,293 rows | 0.459 | 0.420–0.503 | **+0.014 … +0.071** | **1.987** |
| GBM L1 + affine | 2 | 0.461 | 0.417–0.512 | **+0.014 … +0.080** | 2.693 |
| GBM 1-D + scale | 1 | 0.487 | 0.437–0.548 | **+0.045 … +0.104** | 2.776 |
| GBM L1 + scale | 1 | 0.498 | 0.443–0.568 | **+0.049 … +0.130** | 2.800 |
| GBM L1 + dep + scale | 1 | 0.505 | 0.441–0.569 | **+0.053 … +0.131** | 2.756 |
| GBM L1 + quantile map | nonparametric monotone | 0.542 | 0.500–0.593 | **+0.090 … +0.160** | 2.876 |

(The flat height fitted on the train half is 2.3416 m against the shipped 2.3412 m — the
shipped constant is the full-sample median, so it is scored here in its held-out form
rather than in-sample.)

**The two-parameter closed form wins every comparison.** Read the two interval columns
together, because they say different things and only one of them is the test: the marginal
95% intervals overlap heavily — with 1,362 rows in 461 panoramas the *level* of any single
median is uncertain to ±0.04 m — while the **paired** differences, which resample the same
panoramas for both models and difference within the draw, exclude zero for every booster
arm. No arm beats the closed form in more than 0.2% of 2,000 draws. `D rescaled` is the
one row that is not separated, which is Stage 4's own finding that the rescale and the
flat variant are equivalent, reproduced here.

Three of those rows exist specifically to close the escape routes:

- **"It only got one parameter."** The affine arm gets two, fitted under the same absolute
  loss as the metric. It reaches 0.461 m — still behind.
- **"A linear correction is the wrong shape."** The quantile map absorbs *any* monotone
  distortion of the answer scale. It is the worst arm of all (0.542 m), because what it
  cannot fix is which row the booster ranks where — and that is precisely what is wrong.
- **"An era-trained model was never going to transfer."** A booster trained on modern truth
  itself reaches 0.459 m. It is badly underpowered — 1,293 training rows against the era
  split's 316,118 — and it carries a second handicap that is easy to miss: it is fitted
  through `run_gbm_ceiling`'s feature builder, whose `label_type` categorical is the era's
  seven types, so `Crosswalk` and `Signal` (16.3% of these rows) are a missing category in
  its own training data. Both handicaps push the same way, which is why this arm is quoted
  as a **floor** on what modern data supports and never as a modern ceiling. It does not
  clear the closed form either.

![Figure 28 — left: the calibrated comparison with paired cluster-bootstrap intervals; the shipped two-parameter closed form is ahead of every recalibrated booster. Middle: the mechanism, implied camera height by panorama resolution for the era and modern truth sets. Right: median error by true-distance bin — the shipped form leads below 15 m and is last above it, but so is it behind the era blend there, which §8 reads as bias rather than structure.](../figures/fig28-gbm-transfer.png)

## §7 · The mechanism: the era truth is not one scale

`median(truth × tan depression)` — the camera height a truth set implies — cut by the axis
the booster leans on hardest (`sv_norm` + `pano_height` were 13.8% of its split gain in
#6 §6):

| population | no `pano_height` (DC) | 1664 px | 6656 px | 8192 px | pooled |
|---|---:|---:|---:|---:|---:|
| era truth (2017–2020) | **2.802** (n = 223,814) | 2.060 (n = 272) | **2.785** (n = 36,617) | **2.351** (n = 121,177) | 2.636 |
| modern truth (2021+) | – | – | 2.433 (n = 186) | **2.331** (n = 2,302) | 2.341 |

Every group the run kept is in that table — the row counts sum to the pooled n, and a
findings test asserts that they do, because a resolution quietly left out of a table about
heterogeneity would be arguing the point by omission. The 1664-px group is 272 rows (0.07%
of the era frame) and carries no weight in anything below; it is shown because it sits
*lower* than either of the large groups, which widens the spread rather than narrowing it.

The shipped constant is **2.3412 m**. The era truth's 8192-px subpopulation already
implied **2.351 m** — within 1 cm of it, and within 2 cm of the modern measurement. The
"era truth is inflated by ~16%" result from Stage 4 is, on this cut, not a global
inflation at all: it is DC and the 6656-px rigs sitting at ~2.79–2.80 m and carrying 59%
and 9% of the rows respectively, dragging a pooled fit upward.

That is what the ceiling was made of. The era fit pooled these into one per-type height
table, so the era blend answers on the pooled scale everywhere and is wrong in opposite
directions on each subpopulation. A booster handed `pano_height` — including its
*absence*, which is exactly the DC indicator — can answer on each subpopulation's own
scale. Inside the era truth that is worth **0.44 m** (the full booster's margin over the
single-signal one). On a population with a different resolution mix it is worth
**−0.01 m**, because one global factor already absorbs everything it knew.

This refines, rather than contradicts, what was already on the record: [#3
§2](2026-08-07-distance-refit.md) detected the same axis as a significant height-group
level shift (`B_log`'s `log_h` coefficient, −5.617 ± 0.185 — 30 SE from zero) and figure 6
shows the separation. What is new is its size in metres of implied camera height, and the
fact that it — not scene geometry — is what the #6 benchmark was measuring.

**What is not settled here** is *why* the era subpopulations differ. A genuine per-rig
camera-height difference and a truth-pipeline artifact both remain live: #7 §6 measures
2.337–2.559 m across four GSV cities by triangulation, and #3 §6's curb-height and
terrain-model caveats apply to the era truth throughout. This report needs only the weaker
fact — that the scale is *conditional*, and that one number per target population absorbs
it.

## §8 · What does transfer — one clean thing, and one that needs a caveat

**The tail, cleanly.** p90 does not invert anywhere: 3.553 m for the shipped form against
2.800 m for the booster plus a scale, 2.693 m for the affine arm and **1.987 m** for the
modern-trained one. The era blend gets no such benefit (3.603 m), so this is the boosters',
not a by-product of predicting long. They are worse at the typical row and better at the
bad row. A reader who takes only the median from this report takes the wrong half of it.

**The far field, with a caveat that removes most of it.** Median absolute error by
true-distance bin, held-out half:

| bin (m) | n | D era blend | D flat (shipped) | GBM L1 + scale |
|---|---:|---:|---:|---:|
| 2–5 | 168 | 0.612 | **0.169** | 0.242 |
| 5–10 | 590 | 1.136 | **0.316** | 0.348 |
| 10–15 | 336 | 1.786 | **0.475** | 0.671 |
| 15–20 | 130 | **0.769** | 1.789 | 1.275 |
| 20–30 | 80 | **1.984** | 4.561 | 2.607 |
| 30–50 | 54 | 12.297 | 15.812 | **10.470** |

The crossover is sharp and sits at 15 m. Below it — 81% of the rows — the shipped closed
form leads everywhere. Above it the shipped form is the *worst* of the three, and the
booster does beat it in every bin, which is where #6's "the gap widens with distance"
reading would seem to survive.

It mostly does not, and the era blend column is why. That model is biased **+1.07 m long**
and has no conditional structure worth the name, yet it beats the booster at 15–20 m and
20–30 m — because in bins where every bounded model undershoots, a positive bias is
indistinguishable from skill. The booster leads outright only at 30–50 m, on 54 rows. So
the honest reading of the far field is that it is mostly the shipped form's saturation and
its calibrated-short scale, not structure the closed form is failing to use. And it is, as
#3 §6 and #7 §8 both note, exactly where the depth truth is weakest — the least secure
place in the dataset to measure a lead.

**And none of it is the unseen-type handicap.** On the 1,123 held-out rows whose label type
the booster *did* see, the closed form still wins — 0.389 m against 0.478 m — so the 433
`Crosswalk`/`Signal` rows are not carrying the result.

## §9 · What this changes, and what it does not

- **#6's question is now answered twice.** The ceiling exists inside the era truth frame
  (+74% lat/lng, unchanged) and does not exist outside it: with one modern parameter each,
  the shipped two-parameter closed form is **17.6% better** than the booster, not 74%
  worse. The "unmeasurable from inside this dataset" caveat in #6 §5 was the load-bearing
  one, and the answer is that a large share of that ceiling was truth-frame structure.
- **The #3 recommendation stands, and its Stage 4 form is vindicated from a new
  direction.** Dropping the per-type table for one flat height did not merely simplify the
  model — it removed the very degrees of freedom that were absorbing a conditional truth
  scale. The booster's failure to beat it is that decision re-tested with a far more
  flexible model.
- **Ambition for a future refit should be scaled down, and pointed.** There is no 0.4 m of
  general headroom to chase. What is left is the **tail** — a real, model-shaped gap the
  era blend does not share — and, much more weakly, the far field beyond 15 m, where a
  merely long-biased model does about as well as a booster and the depth truth is at its
  softest. If the tail is worth attacking, the cheap experiment is a bounded closed form
  with a better far-field shape, scored on p90 rather than the median; if the far field is,
  the instrument to reach for is better truth
  ([#7](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/7),
  [#20](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/20)), not more
  model capacity.
- **This still must not ship**, for every reason #6 gave, plus one it did not have: the
  headroom it was chasing was partly a property of the data it was trained on.
- **This is not the fully independent test.** Modern truth is Google's depth again, a
  different vintage of the same instrument. It changes the truth *frame* — different
  cameras, different rigs, different resolution mix, no shared fitted parameter — which is
  enough to answer #6. The frame that shares nothing is #7's bearing-only triangulation,
  and its population (auto-labeler detections carrying only a depression angle and a
  panorama height) cannot exercise the booster's conditional inputs at all, which is the
  same reason it cannot host this test.

## Reproducing this report

```bash
pip install -r python/requirements.txt        # includes lightgbm (benchmark-only)
python python/run_gbm_transfer.py --write     # 2-7 min, offline, byte-identical
python python/gbm_transfer_figures.py         # figure 28 (renders the committed summary)
pytest tests/test_gbm_transfer_contract.py tests/test_gbm_transfer_findings.py
```

No network anywhere. The complete input list: the committed era CSVs, the R-fixture split,
`data/modern-truth-labels.csv.gz`, `distance-refit-summary.json` (the era blend coefficients),
and the committed #6 and Stage 4 summaries (the shipped `final_coefficients`, and the
comparability assertions).

---

*Report generated with [Claude Code](https://claude.com/claude-code) — Opus 5 (1M context),
`claude-opus-5[1m]`; review fixes by the same model;
every headline number is asserted by `tests/test_gbm_transfer_findings.py` against
`data/gbm-transfer-summary.json`, which regenerates deterministically from the committed
data.*
