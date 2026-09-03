# label-latlng-estimation

How Project Sidewalk estimates the latitude/longitude of a label from where it was placed in
the GSV viewport. The six per-zoom OLS fits published here (January 2021, Mikey Saugstad) run
on every label placement in every Project Sidewalk city; they replaced Google's depth-data API
after it was withdrawn
([SidewalkWebpage#2374](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/2374)).

In August 2026 the repo grew from a frozen methods record into a full investigation: the lost
input data was reconstructed from production, the analysis was ported to Python, the estimator
was refit as geometry, and the refit was validated against imagery and human clicks it was
never fit on. The result ships as `final_coefficients` in
`data/modern-truth-summary.json` ([#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3)), and has since been checked against an anchor that uses no depth data at all ([#7](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/7)).

**Contents:** [The estimator](#the-published-estimator-unchanged-in-production-since-2021) ·
[Quick start](#quick-start) · [Reports](#reports--the-2026-investigation) ·
[Repository layout](#repository-layout) · [Data & provenance](#data--provenance) ·
[Tests](#tests)

## The published estimator (unchanged in production since 2021)

For zoom z ∈ {1,2,3}:

```
distance       = a_z + b_z·sv_image_y + c_z·canvas_y
heading offset = d_z + e_z·canvas_x
```

Median test error 1.47 m. Coefficients: see the Results section of
`scripts/label-latlng-estimation.md`. Production applies the per-zoom coefficients (not a
pooled fit) — verified during the [modern-truth close-out](reports/2026-08-07-modern-truth.md).

A geodesy footnote uncovered during the port: the R analysis measured distances on a sphere
(`distHaversine`) but computed bearings and destination points on the WGS84 ellipsoid
(`geosphere::bearing`/`destPoint`), while production applies the coefficients with turf's
spherical `destination`. Centimeter-scale, but a refit should pick one model deliberately —
see the notes in `python/label_latlng_estimation.py`. The 2026-08-07 refit picked spherical
(matching production) and scored the switch at −1.8 cm on the median.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate  # see "Python environment" for Windows
pip install -r python/requirements.txt
python python/run_analysis.py        # seven-estimator comparison + coefficients vs published 2021
python python/run_distance_refit.py  # the issue #3 candidate ladder vs the 2021 distance half
python python/run_triangulation.py build --write  # the issue #7 depth-free bearing anchor
                                     #   (~12 min, offline; regenerates data/triangulation-summary.json)
pytest                               # 477 tests — see "Tests" below
```

The R side needs R ≥ 4.x with readr/dplyr/tidyr/tibble/purrr/geosphere/lme4/jsonlite
(pinned in `renv.lock`):

```bash
Rscript scripts/rerun-analysis.R     # regenerates tests/fixtures/r-baseline/
```

### Python environment

Standard venv, no repo-specific tooling. Python ≥ 3.10; developed across macOS, WSL and
native Windows.

```bash
python -m venv .venv
source .venv/bin/activate     # macOS, WSL
.venv\Scripts\Activate.ps1    # Windows PowerShell  (.venv\Scripts\activate.bat for cmd)
pip install -r python/requirements.txt
```

A venv links against one platform's binaries, so a checkout shared between native Windows
and WSL needs one each — `python -m venv .venv-wsl` beside it. Anything matching `.venv*` is
gitignored.

One dependency installs cleanly and then fails to load: **LightGBM's wheels need an OpenMP
runtime pip cannot ship**, and its absence surfaces as a `dlopen` error on
`@rpath/libomp.dylib` (or the platform equivalent) rather than as a failed install.

| platform | fix |
|---|---|
| macOS | `brew install libomp` |
| Debian / Ubuntu / WSL | `sudo apt-get install libgomp1` |
| Windows | ships with the Visual C++ redistributable (x64) |

LightGBM is benchmark-only (issue #6): only `run_gbm_ceiling.py` and `run_gbm_transfer.py`
need it, and everything else — including the rest of the test suite — runs without it.

Two caveats worth knowing before you regenerate anything under `data/`:

- **`requirements.txt` gives lower bounds, not pins.** Fine for reading and for the test
  suite; not a guarantee for regenerating a committed summary. Both LightGBM runners record
  the host and library versions they ran under in their summary's `meta.host`, so a
  regeneration that lands somewhere different is diagnosable. A committed lockfile would make
  a *same-platform* rerun exact and is still worth having; what
  [#22](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/22) measured is that
  no lockfile can make a *cross-platform* one exact — see the next bullet.
- **The LightGBM benchmarks do not reproduce bit-for-bit across hosts — by nature, not by
  neglect.** On an Apple-silicon macOS host in August 2026, refitting the #6 boosters
  reproduced `gbm-ceiling-summary.json`'s `best_iteration` for all four variants and its
  metrics to ~1e-9 for three of them — but `gbm_dep_l1`, the one variant that eats the
  trig-derived `depression_deg`, landed 5.9e-5 m away on the test median. That gap was
  identical across 8 thread counts, two Python versions (3.10, 3.14) and two NumPy/pandas
  stacks, so it is neither: LightGBM picks splits at histogram bin boundaries and a ULP of
  `arctan` is enough to move one. **What changed in
  [#22](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/22):** the guard in
  `run_gbm_transfer.py` used to assert bit-identity (1e-9) before scoring a modern row, which
  turned a 0.059 mm difference into a hard stop and left these two artifacts regenerable on
  one machine. It now asserts the tolerance the findings actually rest on — 1 mm on medians,
  2 cm on p90s, against a 0.40 m headline — stops past that, and *records* whether the run
  was bit-identical instead of demanding it. So `gbm-*-summary.json` regenerates anywhere;
  only its bytes are the recorded host's. Full reasoning in `data/MANIFEST.md`.

## Reports — the 2026 investigation

Dated reports in `reports/` tell the story in order. Start with the first; each builds on the
ones before it.

| Date | Report | What it establishes |
|---|---|---|
| 2026-08-05 | [Recovery & verification](reports/2026-08-05-recovery-and-verification.md) | The lost 2021 dataset reconstructed from production and proven reproduction-grade; the figure evidence behind the refit issues. ([#1](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/1)) |
| 2026-08-05 | [Depth pilot](reports/2026-08-05-depth-pilot.md) | Fresh GSV depth fetched via streetlevel validates the recovered ground truth to the storage floor. ([#4](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/4)) |
| 2026-08-06 | [Depth validation](reports/2026-08-06-depth-validation.md) | The depth is authentic, and it is a *model* — terrain plus building footprints, near-flat-earth under a label. ([#9](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/9)) |
| 2026-08-06 | [Depth coordinate conventions](reports/2026-08-06-depth-coordinate-conventions.md) | Three coordinate frames are in play and they disagree; getting one wrong is silent. **Read this before wiring depth into anything.** |
| 2026-08-06 | [POV inversion](reports/2026-08-06-pov-inversion.md) | The estimator's heading half is exact geometry, zero coefficients — and the 2017–2020 ground truth carries two measured placement artifacts every refit must know about. ([#5](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/5)) |
| 2026-08-07 | [Distance refit](reports/2026-08-07-distance-refit.md) | The distance half refit as geometry: a zero-parameter cotangent already beats all 15 fitted coefficients, the chosen 8-parameter saturating form cuts the median error by a third, and #4765's resolution defect is located in the apply path. ([#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3)) |
| 2026-08-07 | [Mapillary falsification](reports/2026-08-07-mapillary-falsification.md) | The refit scored on imagery it was never fit on: compression gone on both Mapillary cities, most of the height residual traced to per-rig camera heights that transfer to held-out sites, and the deployed model's clovis compression measured at −1.40 m/m. ([#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3) Stage 3) |
| 2026-08-07 | [GBM ceiling](reports/2026-08-07-gbm-ceiling.md) | A LightGBM benchmark on the same split bounds the refit from above: how much accuracy the closed form leaves on the table. ([#6](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/6)) |
| 2026-08-07 | [Modern truth](reports/2026-08-07-modern-truth.md) | The absolute check self-consistency provably could not do: post-2021 human clicks in 49 city schemas against fresh GSV depth. The blend's geometry survives; its *scale* is the era fleet's (a uniform +13%, traced to the era payloads' pinned 2.50 m ground planes), one held-out constant fixes it to 0.41 m median error, and the decision — a single flat 2.34 m height, tradeoffs in §9 — ships as `final_coefficients`. Stored positions are the estimator's own echo in both front-end eras. ([#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3) close-out) |
| 2026-08-08 | [Bearing-only triangulation](reports/2026-08-08-bearing-only-triangulation.md) | The external anchor `final_coefficients` asked for: object positions fixed by the *intersection of bearings*, using no vertical model, no camera height, no depth and no resolution. The ecosystem's assumed 2.6 m camera height is too tall on all six auto-labeler runs; the shipped 2.3412 m is bracketed to ~8% but not confirmed more tightly; and depth and bearings disagree by 13.8% at identical pixels — a multiplicative gap whose shape points at the depth model's scale, not adjudicated absolutely. ([#7](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/7)) |
| 2026-08-10 | [GBM transfer](reports/2026-08-10-gbm-transfer.md) | The ceiling above, re-asked against a second truth frame: it does not survive. With one modern parameter on each side, the shipped two-parameter closed form beats the booster (0.410 m vs 0.498 m) and every richer recalibration of it. The mechanism is that the era truth is not one scale — it implies 2.80 m of camera height at DC and 6656-px panoramas but 2.35 m at 8192 px — so what looked like interaction structure was a booster reading which subpopulation answers on which scale. What does transfer is the tail (p90 3.55 m → 2.80 m for the transferred booster, 1.99 m for one trained on modern truth), and — much more weakly — the far field beyond 15 m. ([#6](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/6)) |
| 2026-09-02 | [Production sign-off](reports/2026-09-02-production-signoff.md) | The estimator as SidewalkWebpage ships it (`approximation3`), signed off for production: 0.40 m vs the regression's 1.08 m on modern truth, winning every zoom, label type, resolution and city (a height calibrated on every *other* city still wins everywhere); on the regression's own 720×480-era split it still edges it (1.38 vs 1.46 m) and beats it by half a metre at equal calibration budget, the gap being the era truth's inflated scale. Geodesy settled (the 6371 km sphere; ≤ 11 cm from WGS84 at the largest answer), a 58-case fixture pinning the Scala, JS and SQL implementations to 1e-9°, and the frame contract Immersive Explore needs. ([SidewalkWebpage#5084](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5084)) |

## Repository layout

| Path | What it is |
|---|---|
| `reports/` | The dated analysis reports above. |
| `python/` | **The authoritative implementation going forward** ([#2](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/2)) — see the module map below. |
| `scripts/label-latlng-estimation.Rmd` (+ `.md`/`.html`) | **The frozen 2021 analysis** — methods record and published coefficients. Kept unmodified. |
| `scripts/rerun-analysis.R` | The Rmd's pipeline as a plain R script; regenerates the R baseline fixtures the tests compare the port against. |
| `scripts/extraction/` | SQL + runners that regenerate the datasets from the production databases: `extract-depth-labels.*` (the frozen 2021 depth population) and `extract-modern-labels.*` (the post-2021 modern-truth sampling frame; city schemas discovered at run time). |
| `data/` | Datasets and committed evidence bundles — see [Data & provenance](#data--provenance) and `data/MANIFEST.md`. |
| `figures/` | The figures referenced by the reports (`python/make_figures.py` and friends regenerate them), plus `triangulation-conclusions.html` — a self-contained interactive page of the issue-#7 findings (`python/triangulation_viz.py build`). |
| `tests/` | Data contract, R↔Python equivalence (~1e-8), and findings-vs-published reproduction tests. |

### Python module map

Each investigation is a library module plus a `run_*.py` entry point that writes its summary
JSON to `data/`:

| Entry point | What it does | Issue |
|---|---|---|
| `run_analysis.py` | Full port of the 2021 analysis: all seven candidate estimators, coefficients vs published. | [#2](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/2) |
| `run_depth_pilot.py` (+ `gsv_depth.py`) | Fetches fresh GSV depth and replicates the 2020 depth→latlng pipeline bit-exactly. | [#4](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/4) |
| `run_depth_validation.py` (+ `depth_validation.py`) | Validates depth authenticity against verbatim GSV imagery: registration, cross-vintage pairs, occlusion adjudication. | [#9](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/9) |
| `verify_depth_conventions.py` | **Reproduces every coordinate-convention finding in one command** (checks A–G), offline. Locked by `tests/test_depth_conventions.py`. | [#9](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/9) |
| `run_pov_inversion.py` (+ `pov_inversion.py`) | Exact click→POV inversion (verified against production to ≤1 px) and the era-faithful forward model of the 2017–2020 placement pipeline. | [#5](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/5) |
| `run_distance_refit.py` (+ `distance_refit.py`) | The distance-half refit: the geometry-shaped candidate ladder, both losses, Stage-2 robustness scoring. | [#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3) |
| `run_mapillary_falsification.py` (+ `mapillary_falsification.py`) | Stage 3 falsification: Mapillary metadata census, #4766's scale-free diagnostics reimplemented, per-sequence camera heights. | [#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3) |
| `run_gbm_ceiling.py` | Benchmark-only LightGBM accuracy ceiling on the refit's split. | [#6](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/6) |
| `run_gbm_transfer.py` (+ `gbm_transfer.py`) | Scores those same boosters against modern-truth rows they were never fitted on, to separate scene structure from era-truth-frame structure. | [#6](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/6) |
| `run_modern_truth.py` (+ `modern_truth.py`) | The absolute close-out: stratified modern-label sample, heading-centred depth lookup (shared bit-for-bit with the legacy path via `depth_validation.classify_depth_pixel`), era-aware circularity guard, held-out remedy check. | [#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3) |
| `run_triangulation.py` (+ `triangulation.py`, `triangulation_depth.py`) | Bearing-only triangulation: leave-one-out ray intersection as a depth-free range truth, its error budget and bias validation, and the same-pixel depth cross-check. | [#7](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/7) |
| `run_signoff.py` (+ `signoff.py`, `signoff_figures.py`) | Production sign-off of the shipped estimator: both truth frames head-to-head with the 2021 regression, hold-out and leave-one-city-out calibration checks, geodesy, the viewport frame contract, the cross-implementation parity fixture, and the worked examples. | [SidewalkWebpage#5084](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5084) |

## Data & provenance

The 2021 input CSVs were gitignored and lost; the committed data is a **reconstruction** from
the production databases, regenerated 2026-08-05
([#1](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/1)). The depth-label
population turned out to be frozen since 2021, so it is effectively a reproduction: identical
cleaned counts (395,147 total; 316,118 train / 79,029 test) and matching findings. Full
provenance, caveats, and regeneration instructions: **`data/MANIFEST.md`**.

| Files | Contents |
|---|---|
| `data/labels-*-latlng.csv.gz` | The dataset: depth-derived ground-truth label positions for 7 cities. |
| `data/depth-pilot-*` | Depth pilot evidence ([#4](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/4)): raw depth payloads for 409 panos, the per-label cross-check, per-pano camera heights. |
| `data/depth-validation-*` | Depth validation evidence ([#9](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/9)): **verbatim GSV imagery tiles** for 60 panoramas, per-panorama yaw and road-link bearings, registration scores, label hits, cross-vintage pairs, and the hand occlusion adjudication. Everything after `fetch` replays these bytes offline. |
| `data/falsification-*` | Stage 3 inputs ([#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3)): the auto-labeler's fused multi-view curb-ramp sites and per-pano Mapillary/GSV metadata for six cities — gitignored artifacts in their home repo, preserved here. |
| `data/modern-truth-*` | Modern-truth evidence ([#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3)): verbatim depth payloads for 1,106 modern panos, per-pano fetch status + camera-height QC, and per-label truth/gates/predictions/guard for 3,286 post-2021 labels. Everything after `fetch` replays these bytes offline. |
| `data/triangulation-*` | Bearing-only triangulation evidence ([#7](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/7)): the summary over six auto-labeler runs, plus verbatim depth payloads for 480 GSV panoramas and their fetch metadata for the same-pixel anchor. Everything after `fetch` replays these bytes offline. |
| `data/*-summary.json` | Each investigation's machine-readable results, including `final_coefficients` in `modern-truth-summary.json`. |

## Tests

`pytest` runs 477 tests: data contract, R↔Python equivalence, findings-vs-published, depth
pilot, depth validation, coordinate conventions, POV inversion, distance refit (findings +
invariants), Mapillary falsification, GBM ceiling, modern truth (findings + invariants),
bearing-only triangulation (estimator invariants on known geometry + findings + the
conclusions-page build), and GBM transfer (frame-mapping contract + the frozen-booster
guard + findings).

`RUN_SLOW=1 pytest` additionally re-derives the coordinate-conventions evidence in full from
the committed bytes, re-reads every modern-truth label's truth from its payload, rebuilds
the issue-#7 conclusions page byte-for-byte from the committed bundles, and recomputes the
era-frame pixel residuals behind the #6 transfer report's frame-mapping table.
