# Modern truth: the blend's geometry is right, its scale is the old fleet's — and one constant fixes it

**2026-08-07** · issue [#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3) (the absolute-scale close-out) · scores the candidate shipped by [Stages 1–2](2026-08-07-distance-refit.md) after the [Mapillary falsification](2026-08-07-mapillary-falsification.md) · feeds [SidewalkWebpage#4765](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4765) and [#4766](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4766)

| | |
|---|---|
| **0.43 m** | held-out median distance error of the shipped blend after ONE global height rescale (×0.863), against fresh-depth truth on post-2021 human labels — vs 1.24 m for the deployed model on the same population |
| **+1.07 m** | the shipped blend's signed bias on modern truth — uniform across all 13 scoreable cities (+0.4 to +2.0) and every capture year: a pure scale error, not a shape error |
| **2.35 m** | the measured modern camera height — and the effective height every one of the nine label types' modern truth implies (2.27–2.37, flat). The fitted 2.50–2.78 m table is the era truth's pinned-plane scale plus its terrain bias, not click geometry |
| **+1.77 m** | what SidewalkWebpage#4765's one-line normalization alone does on modern truth — the refit predicted +1.70 from era data alone; the one-liner and the refit still have to travel together |
| **97.6% / 75.6%** | of stored post-2021 positions that reproduce from their own era's formula (±0.5 m; real-pixels / fixed-frame eras) — stored lat/lng is the estimator's own echo everywhere, and any "validation" against it is self-grading |

> Reproduce every number here (offline, from the committed artifacts):
>
> ```bash
> python python/run_modern_truth.py build --write   # cache/artifacts -> summary (~15 min)
> python python/modern_truth_figures.py             # figures 20-23
> pytest tests/test_modern_truth_findings.py        # the findings, locked
> ```
>
> Only `run_modern_truth.py fetch` touches the network; its 1,106 depth payloads are
> committed verbatim in `data/modern-truth-payloads.jsonl.gz`, so everything above replays
> from a fresh checkout.

## §1 · Goal

The Mapillary falsification certified the blend's *shape* with scale-free diagnostics and said
plainly what it could not do: self-consistency provably cannot see a shared scale error
(its §8). This report supplies the missing **absolute** check, on the population every prior
stage lacked: **real human clicks, placed by the modern front end, on modern imagery, across
every current GSV city** — scored against ground truth that owes nothing to any estimator:
fresh GSV depth sampled at the stored click pixel.

Three facts make that population uniquely clean. Post-evolution-179 `pano_x/pano_y` replays
the front-end projection at 100.0000% (`data/pov-inversion-summary.json`), so the stored pixel
is a trustworthy anchor. Both that pixel and the depth raster are heading-centred
(the [conventions report](2026-08-06-depth-coordinate-conventions.md) §1), so the lookup needs
no mirror and no yaw rotation — the open-item-G ambiguity that clouds the era ground truth
does not exist here. And no era heading-bias correction applies. **No parameter is fitted
anywhere in the main comparison**: blend D is scored exactly as committed in
`data/distance-refit-summary.json`.

## §2 · Questions

- **Q1 — Is the depth lookup's frame right?** → **§4**: proven three ways — a synthetic
  payload where the cotangent is exact, a pixel-level cross-check on the 409 committed pilot
  payloads (row agreement 100%, mirrored formula agrees *nowhere*), and wrong-frame controls
  on the modern data itself (fig 23).
- **Q2 — Are stored positions usable as truth?** → **§5**: no, and now it's measured: they are
  estimator echoes in both eras — with a previously undocumented **2 m-scale era
  discontinuity** at evolution 179, plus the discovery that production applies **per-zoom**
  coefficients (`PanoDataService.scala`), not the zoom-1 triple alone.
- **Q3 — The deployed model against absolute truth?** → **§6**: median |err| 1.24 m,
  p90 5.74 m, range slope −0.45 m/m — #4766's compression, reproduced against ground truth
  instead of self-consistency. Its small headline bias (−0.28 m) is the known two-error
  cancellation on 8192-px panos, not calibration.
- **Q4 — Would #4765's one-liner alone have sufficed?** → **§6**: no — +1.77 m bias on modern
  truth, the overcorrection the era analysis predicted at +1.70. The normalization is only
  correct *inside* the refit.
- **Q5 — The blend against absolute truth?** → **§6–§7**: the shape survives (flat bias curve
  where the linear models dive, best p90 at 4.33 m), but the whole curve sits **+1.07 m
  high — a uniform ~13% scale error** across every city and capture year.
- **Q6 — Where does the +13% come from?** → **§7**: the era ground truth's payload
  generation. 68% of era payloads pin the ground plane at exactly 2.50 m (a default, not a
  measurement); modern payloads mostly measure it (median 2.35 m). Within this one dataset,
  pinned-plane panos imply 2.42 m and measured-plane panos 2.28 m. The blend's fitted heights
  absorbed the era scale; the per-type spread (0.28 m) **does not replicate** — all nine types
  imply the same 2.27–2.37 m (fig 22).
- **Q7 — Do the never-fitted types survive `height_fallback_m`?** → **§7**: yes — Crosswalk
  (n=322) and Signal (n=64) imply 2.37/2.30 m, indistinguishable from the fitted seven; the
  fallback rule is exactly as wrong as the rest of the table, no worse.
- **Q8 — Real near-horizon clicks?** → **§8**: 0.9% of human labels sit at ≤2° depression,
  where truth is far and every bounded model undershoots; the blend's clamp is the least-bad
  answer (median −12.2 m vs the deployed −15.8 m), and the raw cotangent is unusable (+12.4 m
  the other way).
- **Q9 — What fixes the scale?** → **§9**: one constant. A global height rescale (k = 0.863,
  fitted on half the panos, scored on the disjoint half) takes the blend from 1.27 m to
  **0.43 m** median and +1.09 to −0.18 m bias; a single flat height (2.33 m) does the same
  (0.42 m) — the per-type table earns nothing on modern truth.

## §3 · Dataset

**Labels.** A new read-only extraction (`scripts/extraction/extract-modern-labels.{sh,sql}`,
run against production 2026-08-07) of every post-2021, non-deleted, non-tutorial label with
stored `pano_x/pano_y` on a GSV-sourced pano: **1,206,523 rows across 45 city schemas** —
the seven 2021 cities plus 38 the estimator has since been deployed to. Excluded with
reasons: richmond (Mapillary), the infra3d schemas, the validation-study schema, and two
superseded/backup schemas. Two provenance facts pinned during discovery: every non-tutorial
post-2021 row is `computation_method = 'approximation2'` (the only post-2021 'depth' rows are
tutorial labels on the fixed legacy pano), and AI-submitted labels are exactly the
`SidewalkAI` user's (64,814 rows, all vancouver) — carried as an `is_ai` flag. The raw
extraction is regenerable and stays uncommitted; sampling-frame gates (in-raster pixels,
finite stored coordinates, the 2021 cleaning's ≤20-labels-per-pano guard) leave 1,135,427
rows / 486,547 panos.

**Sampling and fetch.** Stratified pano selection (seed 666, each stratum oversampled 2×):
700 *representative* panos (uniform over human panos, 150-per-city cap — the headline
population), 200 *near-horizon* panos (≥1 label at ≤2°), rarest-first per-type top-ups to
200 labels/type (the first real exercise of the never-fitted types), and 100 SidewalkAI
panos. The fetch walked 1,911 candidates: **1,106 still resolve and serve depth** (58%
resolve — attrition is almost entirely id-gone panos, concentrated in older captures; 99.5%
of resolved panos serve depth; 6 payloads are malformed for streetlevel's parser). Every
stratum met its full budget. 65% of fetched panos are 2022+ captures; fetched origins sit
0.24 m median (p90 0.69 m) from the extraction's `pano_data` origins, so re-registration
drift is a minor term here.

**Truth.** For each label, the depth raster cell is
`col = round(pano_x/width·512) % 512`, `row = clamp(round(pano_y/height·256))` — no mirror
against `gsv_depth`'s payload-order arrays (the conventions report's `511 − …` recipe is for
streetlevel's mirrored output), no yaw shift. Truth is the ray's horizontal ground distance
(`depth_validation.classify_depth_pixel`, the refactored core the legacy v6 path now shares
bit-for-bit) — camera-relative, so immune to origin drift, and independent of camera height.
Gates: ray lands on ground/terrain (facade/sky/oblique excluded — 390 rows, 11.7%), 3×3
neighbourhood agreement (0 further exclusions), truth finite and under the 50 m cap (46).
**3,332 labels on fetched panos → 2,896 scored (2,690 human, 206 AI).**

Committed artifacts: `data/modern-truth-payloads.jsonl.gz` (verbatim base64, 5.3 MB),
`modern-truth-panos.csv.gz`, `modern-truth-labels.csv.gz`, `modern-truth-summary.json`;
the fetch cache is gitignored. Provenance in `data/MANIFEST.md`.

## §4 · The frame, proven three ways

The one genuinely dangerous step is reading a depth map at a stored pixel — the pilot showed
wrong frames produce 7–17 m medians. Three independent checks, all locked by tests:

1. **Synthetic exactness** (`test_modern_truth_contract.py`): on a constructed payload with
   one ground plane at height *h*, the sampler returns `h/tan(depression)` to float
   precision, and the mirrored formula provably reads a different world.
2. **The 409-payload cross-check** (`test_modern_truth_findings.py`): recovered-era labels
   carry both the legacy north-referenced pixel and the evolution-179 heading-centred
   recompute. On the committed pilot payloads, the modern lookup and the yaw-corrected era
   lookup land on the same raster row 100% of the time and the same-or-adjacent column ~84%
   (93.5% within two; the residual tail is the documented legacy camera-heading drift, and
   the median offset is the era `ceil` vs modern `round` half-pixel). The **mirrored column
   formula agrees exactly nowhere** (0% within two columns).
3. **Wrong-frame controls on this dataset** (fig 23): re-reading every label under the
   pilot's null hypotheses annihilates the vertical conventions (1% of rays still hit
   ground; 9.7–13.0 m median errors). The x-mirror is only weakly separated by ground
   *distance* — a road is nearly left-right symmetric in range — which is precisely why
   check 2 exists.

![Figure 23 — frame controls on the modern set: identity keeps 88% of rays on ground/terrain at 1.35 m blend median error; both vertical-flip conventions collapse to 1% ground share and ~10 m errors; the x-mirror is weakly separated by distance alone and is instead rejected outright by the pixel-level cross-check.](../figures/fig23-modern-truth-frame-controls.png)

## §5 · The circularity guard — and two things it found

Stored post-2021 `lat/lng` had to be disqualified as truth *by measurement*, and the
recompute needed to be exact. Getting it exact surfaced two undocumented facts:

- **Production applies per-zoom coefficients.** `PanoDataService.toLatLng` selects one of
  three published triples by `round(zoom)`; a zoom-1-only recompute (all the prior analyses'
  convention, correct for the auto-labeler) leaves 1.0/2.2 m median residuals on the zoom-2/3
  populations. `modern_truth.DEPLOYED_DIST_COEF` now carries all three, verbatim from the
  Scala source.
- **Stored positions have an era discontinuity at evolution 179 (2023-03-29).** Labels placed
  before it were computed by the old front end from **fixed-frame** `sv_image_y` — the
  coefficients' own frame, i.e. *without* the #4765 apply-path defect. Labels after it feed
  real pixels into fixed-frame coefficients — the defect population. Each era reproduces from
  its own formula and not the other's: fixed-frame era median |diff| 0.29 m (wrong era:
  1.96 m), real-pixels era 0.10 m (wrong era: 1.43 m); 97.6% of real-pixels rows sit within
  0.5 m of their recompute (75.6% for the older era, whose panos have had longer to drift).

The consequence for everything downstream: stored positions are the estimator's own output
throughout — a database-wide echo — so the deployed model's predictions here are always the
**recomputed** apply path, and no comparison in this report touches stored coordinates except
this guard.

## §6 · The absolute comparison

Median signed/absolute error against depth truth, representative human stratum (n=1,502;
the pooled-human column, n=2,690, includes the deliberately oversampled near-horizon and
rare-type strata):

| model | median \|err\| | signed median | p90 \|err\| | range slope (m/m) |
|---|---:|---:|---:|---:|
| A deployed (per-zoom, real px) | 1.10 | −0.28 | 3.86 | −0.31 |
| B #4765-normalized only | 2.08 | +1.77 | 4.02 | −0.38 |
| C 2.6 m cotangent | 1.04 | +1.02 | 3.44 | +0.15 |
| D blend as shipped | 1.20 | +1.07 | 3.23 | −0.23 |

![Figure 20 — predicted vs true distance: the deployed model bends across the diagonal (compression), the shipped blend tracks it at ~13% high, and the held-out global rescale puts it on the diagonal at 0.43 m median.](../figures/fig20-modern-truth-pred-vs-truth.png)

Reading it (fig 21 shows the same by viewing angle):

- **The deployed model's compression is now measured against ground truth**: too far in the
  near field, increasingly too near beyond ~12 m (−0.45 m/m pooled; p90 5.74 m), ending in
  the −15.8 m median undershoot at ≤2°. Its flattering headline bias is the era report's
  two-error cancellation on 8192-px panos, observed almost exactly (−0.28 m here vs −0.39
  predicted); on any other resolution it has no such luck.
- **B confirms the standing warning numerically**: era analysis predicted the bare one-liner
  would overcorrect to +1.70 m; modern truth says +1.77 m. #4765 must not ship without the
  refit.
- **The blend's *form* does what it was designed to do** — flat bias where both linear models
  dive (fig 21), best p90, bounded near-horizon behaviour — but the whole curve floats
  ~+1.07 m, and the float is *uniform*: +0.4 to +2.0 m across all 13 scoreable cities,
  +0.8 to +1.4 across every capture year, +1.9 on the AI stratum. A shape error varies with
  geometry; a scale error doesn't. This is a scale error.

![Figure 21 — median signed error vs depression: A and B inherit the linear form's angle-dependent bias, C diverges toward the horizon, D is flat — but flat at +1, not at 0.](../figures/fig21-modern-truth-error-vs-depression.png)

## §7 · Where the +13% lives: the era truth's payload generation

The blend takes exactly one physical input family — per-type camera heights. Modern truth
lets each type vote on its own effective height, `median(truth · tan(depression))`:

![Figure 22 — implied effective camera height by label type: all nine types land in the measured rig band (2.35 m), flat; the fitted table sits 0.15–0.45 m higher.](../figures/fig22-modern-truth-implied-heights.png)

Three mutually consistent measurements pin the mechanism:

- **The pin.** Google's payloads either *measure* the ground plane or *pin* it at exactly
  2.50 m (a default the pilot established structurally). Era payloads: 68% pinned. This
  fetch: 31% pinned overall, ~90% for pre-2017 captures, ~25% for 2018+.
- **The split.** Within this one dataset, labels on pinned-plane panos imply 2.42 m; labels
  on measured-plane panos imply 2.28 m. The truth's scale tracks the payload generation.
- **The consistency check.** Crosswalk labels are road paint — zero height offset, never
  fitted, immune to the era table. Their implied height is 2.367 m; the independently
  measured rig height median is 2.354 m. Thirteen millimetres apart.

So the era ground truth the blend was fitted on — 395k labels whose depth came from the
pinned-heavy payload generation — carries a ~+6% default-plane scale, plus the terrain-model
overshoot that report documented (~0.5 m on curb ramps). The fitted heights absorbed both;
that was the *correct* fit to that truth. Modern payloads measure the plane instead, and the
curb overshoot has largely vanished too: correcting modern CurbRamp truth by the classic
`curb·d/h` term now *worsens* the blend's bias (+1.50 → +2.13 m), i.e. the modern terrain
model already hugs the ramp. And with both era artifacts gone, **the per-type spread goes
with them**: all nine types, including the two scored through `height_fallback_m`, imply the
same rig.

## §8 · Real near-horizon clicks

The era analysis could only characterize the near-horizon regime on era clicks; the
falsification's fuse gate excluded it entirely. Here it is on modern human labels (median
signed error, pooled human):

| depression | n | A deployed | C cotangent | D blend |
|---|---:|---:|---:|---:|
| ≤ 0° (above horizon) | 6 | −0.4 | +26.1 | +3.6 |
| 0–2° | 21 | −15.8 | +12.4 | −12.2 |
| 2–5° | 146 | −13.9 | +5.6 | −8.9 |
| 5–11.25° | 824 | −1.1 | +2.0 | +1.5 |
| > 11.25° | 1,693 | −0.1 | +0.7 | +1.1 |

0.9% of human labels sit at ≤2°, 6% under 5° — mostly genuinely distant things (signals,
road ends). There, truth routinely exceeds any bounded answer: every capped model
undershoots, the blend least badly, and the raw cotangent is unusable in both directions.
The blend's clamp policy (≈28 m max) remains the right trade; nothing here argues for
revisiting it.

## §9 · One constant — the held-out remedy check

Because the gap is a uniform scale, the candidate fixes are one-parameter. Fitted on a
random half of the panos, scored on the disjoint half (`modern_truth.remedy_check`; the
falsification's held-out discipline), human gated rows:

| candidate | median \|err\| | signed median | p90 |
|---|---:|---:|---:|
| D as shipped | 1.27 | +1.09 | 3.83 |
| D × k, whole height table rescaled (k = 0.863) | **0.43** | −0.18 | 3.61 |
| D flat, one height for every type (2.33 m) | 0.42 | −0.20 | 3.70 |

The rescale and the flat variant are statistically indistinguishable here — which is itself
the finding: **the per-type table buys nothing on modern truth** (it bought 4 cm of era-test
median: 0.9335 vs 0.9741). The residual −0.18 m and the surviving p90 tail are the honest
remainder: near-horizon undershoot plus whatever the modern terrain model still overstates.

**What this means for the Stage 4 coefficients — a decision, deliberately not made here.**
The evidence says the blend's functional form ships as-is, and its height scale should be
the modern measured one; the two concrete options are (a) multiply the committed height
table and fallback by 0.863, or (b) replace the table with a single 2.33 m height (simpler,
same accuracy, abandons structure modern truth cannot support — at a known 4 cm era-side
cost). One honest limit travels with either: "modern truth" is still Google's own depth
model, now with measured rather than defaulted planes; its internal consistency is strong
(the 13 mm Crosswalk check), but no external geodetic reference has been consulted —
bearing-only triangulation (#7) remains the independent path if one is wanted. The
SidewalkWebpage integration currently in flight consumes the provisional (unrescaled)
coefficients; it should not merge ahead of this call.

## §10 · Deliberately not claimed

- **An absolute verdict beyond GSV's own geometry.** Everything here is calibrated to
  Google's depth planes — measured ones, cross-checked internally, but Google's nonetheless.
- **Mapillary absolute scale.** Unchanged from the falsification's §8: no depth exists
  there; the per-source-height recipe plus the panorama-tools depth store remain the route.
- **AI-label behaviour in general.** The AI stratum is one city, one importer (n=206); its
  larger bias (+1.85 m) is confounded with vancouver's rig and is noted, not interpreted.
- **A backfill recommendation.** Stored positions carry the era discontinuity (§5) and both
  eras' model biases; recomputing them under the final coefficients is feasible from stored
  pixels and is a separate product decision, unchanged from the Stage 4 note.
