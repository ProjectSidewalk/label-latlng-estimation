# label-latlng-estimation

How Project Sidewalk estimates the latitude/longitude of a label from where it was placed in the
GSV viewport. The six per-zoom OLS fits published here (January 2021, Mikey Saugstad) run on
every label placement in every Project Sidewalk city; they replaced Google's depth-data API
after it was withdrawn ([SidewalkWebpage#2374](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/2374)).

## What's here

| path | what it is |
|---|---|
| `reports/` | **Dated analysis reports.** Start with [2026-08-05 — recovery & verification](reports/2026-08-05-recovery-and-verification.md): what was recovered, the proof it's right, and the figure evidence behind the refit issues. Then [2026-08-05 — depth pilot](reports/2026-08-05-depth-pilot.md): fresh GSV depth via streetlevel validates the recovered ground truth to the storage floor ([#4](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/4)). Then [2026-08-06 — depth validation](reports/2026-08-06-depth-validation.md): the depth is authentic, and it is a *model* — terrain plus building footprints, near-flat-earth under a label ([#9](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/9)); and its companion [2026-08-06 — coordinate conventions](reports/2026-08-06-depth-coordinate-conventions.md), which is the one to read before wiring depth into anything. Then [2026-08-06 — POV inversion](reports/2026-08-06-pov-inversion.md): the estimator's heading half is exact geometry, zero coefficients — and the 2017–2020 ground truth carries two measured placement artifacts every refit must know about ([#5](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/5)). Then [2026-08-07 — distance refit](reports/2026-08-07-distance-refit.md): the distance half refit as geometry — a zero-parameter cotangent already beats all 15 fitted coefficients, the chosen 8-parameter saturating form cuts the median error by a third, and #4765's resolution defect is located in the apply path ([#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3)). Then [2026-08-07 — Mapillary falsification](reports/2026-08-07-mapillary-falsification.md): the refit scored on the imagery it was never fit on — compression gone on both Mapillary cities, most of the height residual traced to per-rig camera heights that transfer to held-out sites, and the deployed model's clovis compression measured at −1.40 m/m ([#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3) Stage 3). Bounding the same refit from above, [2026-08-07 — GBM ceiling](reports/2026-08-07-gbm-ceiling.md): a LightGBM benchmark on the same split measures how much accuracy the closed form leaves on the table ([#6](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/6)). The arc closes with [2026-08-07 — modern truth](reports/2026-08-07-modern-truth.md): the absolute check self-consistency provably could not do — post-2021 human clicks in 49 city schemas against fresh GSV depth. The blend's geometry survives; its *scale* is the era fleet's (a uniform +13%, traced to the era payloads' pinned 2.50 m ground planes), one held-out global constant fixes it to 0.41 m median error, stored positions are the estimator's own echo in both front-end eras, and production turns out to apply per-zoom coefficients ([#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3) close-out). |
| `scripts/label-latlng-estimation.Rmd` (+ `.md`/`.html`) | **The frozen 2021 analysis** — methods record and published coefficients. Kept unmodified. |
| `data/labels-*-latlng.csv.gz` | The dataset: depth-derived ground-truth label positions for 7 cities, **reconstructed from production 2026-08-05** ([#1](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/1)). See `data/MANIFEST.md` — the reconstruction reproduces the published row counts and findings exactly. |
| `scripts/extraction/` | The SQL + runners that regenerate the datasets from the production databases: `extract-depth-labels.*` for the frozen 2021 depth population, `extract-modern-labels.*` for the post-2021 modern-truth sampling frame (all current city schemas, discovered at run time). |
| `python/` | **The authoritative implementation going forward** ([#2](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/2)): full port of the analysis (all seven candidate estimators). `gsv_depth.py` + `run_depth_pilot.py` add the issue #4 depth pilot: fetching fresh GSV depth and replicating the 2020 depth→latlng pipeline bit-exactly. `pov_inversion.py` + `run_pov_inversion.py` add the issue #5 exact click→POV inversion (verified against production to ≤1 px) and the era-faithful forward model of the 2017–2020 placement pipeline. `distance_refit.py` + `run_distance_refit.py` add the issue #3 distance-half refit: the geometry-shaped candidate ladder, both losses, and the Stage-2 robustness scoring, written to `data/distance-refit-summary.json`. `mapillary_falsification.py` + `run_mapillary_falsification.py` add the Stage 3 falsification: the Mapillary metadata census, #4766's scale-free diagnostics reimplemented, and per-sequence camera heights, written to `data/falsification-summary.json`. `modern_truth.py` + `run_modern_truth.py` add the absolute close-out: the stratified modern-label sample, the heading-centred depth lookup (shared bit-for-bit with the legacy path via `depth_validation.classify_depth_pixel`), the era-aware circularity guard, and the held-out remedy check, written to `data/modern-truth-summary.json`. |
| `data/depth-pilot-*` | The depth pilot's committed evidence ([#4](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/4)): raw depth payloads for 409 panos, the per-label cross-check, per-pano camera heights. See `data/MANIFEST.md`. |
| `data/depth-validation-*` | The depth validation's committed evidence ([#9](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/9)): **verbatim GSV imagery tiles** for 60 panoramas, per-panorama yaw and road-link bearings, registration scores, label hits, cross-vintage pairs, and the hand occlusion adjudication. Everything after `fetch` replays these bytes offline. |
| `data/falsification-*` | The Stage 3 inputs ([#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3)): the auto-labeler's fused multi-view curb-ramp sites and per-pano Mapillary/GSV metadata for six cities — gitignored artifacts in their home repo, preserved here. See `data/MANIFEST.md`. |
| `data/modern-truth-*` | The absolute close-out's committed evidence ([#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3)): verbatim depth payloads for 1,106 modern panos, per-pano fetch status + camera-height QC, and per-label truth/gates/predictions/guard for 3,286 post-2021 labels. Everything after `fetch` replays these bytes offline. See `data/MANIFEST.md`. |
| `python/verify_depth_conventions.py` | **Reproduces every coordinate-convention finding in one command** (checks A–G), offline. Three frames are in play and they disagree; getting one wrong is silent. Locked by `tests/test_depth_conventions.py`. |
| `scripts/rerun-analysis.R` | The Rmd's pipeline as a plain R script; regenerates the R baseline fixtures the tests compare the port against. |
| `tests/` | Data contract, R↔Python equivalence (~1e-8), and findings-vs-published reproduction tests. |

## Running it

```bash
pip install -r python/requirements.txt
python python/run_analysis.py        # seven-estimator comparison + coefficients vs published 2021
python python/run_distance_refit.py  # the issue #3 candidate ladder vs the 2021 distance half
pytest                               # 356 tests: data contract, R↔Python equivalence, findings,
                                     #            depth pilot, depth validation, coordinate
                                     #            conventions, POV inversion, distance refit
                                     #            (findings + invariants), Mapillary falsification,
                                     #            GBM ceiling, modern truth (findings + invariants)
                                     #            (RUN_SLOW=1 unlocks a full re-derivation of the
                                     #             conventions evidence from committed bytes)
```

The R side needs R ≥ 4.x with readr/dplyr/tidyr/tibble/purrr/geosphere/lme4/jsonlite:
`Rscript scripts/rerun-analysis.R` (regenerates `tests/fixtures/r-baseline/`).

## The published estimator (unchanged since 2021)

For zoom z ∈ {1,2,3}: `distance = a_z + b_z·sv_image_y + c_z·canvas_y`,
`heading offset = d_z + e_z·canvas_x`; median test error 1.47 m. Coefficients: see the
Results section of `scripts/label-latlng-estimation.md`.

A geodesy footnote uncovered during the port: the R analysis measured distances on a sphere
(`distHaversine`) but computed bearings and destination points on the WGS84 ellipsoid
(`geosphere::bearing`/`destPoint`), while production applies the coefficients with turf's
spherical `destination`. Centimeter-scale, but a refit should pick one model deliberately —
see the notes in `python/label_latlng_estimation.py`. The 2026-08-07 refit picked spherical
(matching production) and scored the switch at −1.8 cm on the median.

## Provenance & reproducibility

The 2021 input CSVs were gitignored and lost; the committed data is a **reconstruction** from
the production databases (the depth-label population turned out to be frozen since 2021, so it
is effectively a reproduction: identical cleaned counts 395,147 / 316,118 / 79,029 and matching
findings). Full provenance, caveats, and regeneration instructions: `data/MANIFEST.md`.
