# Dataset manifest — depth-derived label positions (RECONSTRUCTION)

**These are not the original 2021 CSVs.** The files consumed by the 2021 analysis were never
committed and no longer exist ([issue #1](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/1)).
The `labels-*-latlng.csv.gz` files here were **regenerated from production on 2026-08-05** using
`scripts/extraction/`. That said, the reconstruction turned out to be reproduction-grade: the
depth-label population has been effectively frozen in production since 2021, and re-running the
2021 cleaning pipeline on these files yields **exactly** the published row counts
(395,147 cleaned; 316,118 train / 79,029 test) and reproduces the published findings — see
`tests/` and the R baseline in `tests/fixtures/r-baseline/`.

## Extraction provenance

- **Date:** 2026-08-05 (UTC), host `makelab1.cs.washington.edu`
- **Databases:** `sidewalk_prod` (PostgreSQL 16, schema per city, Play evolutions applied: 340)
  for the six modern cities; `sidewalk_dc` (the retired legacy DC database, still at Play
  evolution 19 with the pre-evolution-179 schema) for DC
- **Queries:** `scripts/extraction/extract-depth-labels-modern.sql` (six cities) and
  `extract-depth-labels-legacy-dc.sql` (DC — near-verbatim the 2021 Rmd's documented DC query);
  raw psql session details in `data/extraction-metadata.txt`
- **Selection:** `computation_method = 'depth'` (modern) / `lat IS NOT NULL` (legacy DC). The
  2021 extraction was unfiltered, but every non-depth row was discarded by the analysis'
  cleaning filters, so nothing the analysis uses is lost.

## Row counts

| city | source | rows | rows with `time_created` ≥ 2021-01-01 |
|---|---|---|---|
| dc | `sidewalk_dc` (legacy schema, evolution 19) | 270,845 | 61 (plus 82,791 NULL — early rows predate the column) |
| seattle | `sidewalk_prod.sidewalk_seattle` | 120,094 | 3,808 |
| newberg | `sidewalk_prod.sidewalk_newberg` | 17,725 | 38 |
| columbus | `sidewalk_prod.sidewalk_columbus` | 20,526 | 395 |
| spgg | `sidewalk_prod.sidewalk_spgg` | 31,498 | 792 |
| cdmx | `sidewalk_prod.sidewalk_cdmx` | 6,772 | 100 |
| pittsburgh | `sidewalk_prod.sidewalk_pittsburgh` | 1,148 | 412 |
| **total** | | **468,608** | |

Verified at extraction time: every depth row in the six modern cities has an
`old_label_metadata` record (0 missing across all cities), so the pre-evolution-179 values of
`sv_image_x/y`, `photographer_heading/pitch`, and `panorama_lat/lng` are recovered exactly.

## Columns

Columns 1–22 are **exactly the 2021 column set** (names, order, and semantics of the original
extraction — see the mapping comments in `scripts/extraction/extract-depth-labels-modern.sql`).
Columns 23–27 are extras that did not exist in the 2021 CSVs, appended so both `readr` and
`pandas` read the files unchanged:

- `pano_width`, `pano_height` — panorama pixel dimensions from today's `pano_data` (pano
  resolution never changes for a given pano id, so these are valid for when the label was
  placed; NULL where GSV no longer serves the pano — e.g. 2,017 Seattle rows)
- `time_created` — label creation time (NULL for early DC rows that predate the column)
- `current_pano_x`, `current_pano_y` — the post-evolution-179 *recomputed* pano coordinates,
  for cross-checking against the original `sv_image_x/y` (NULL for DC: evolution 179 never ran
  on the legacy database)

## Caveats

1. **The `time_created < 2021-01-01` cutoff matters for faithfulness.** Evolution 93 stamped
   `computation_method = 'depth'` on every row that had lat/lng when it ran (mid-2021), so this
   export contains 5,606 rows created after the 2021-01-01 extraction. Both analysis pipelines
   (`scripts/rerun-analysis.R`, `python/`) filter them out (keeping NULL `time_created`, which
   marks pre-column DC rows) — that is what makes the cleaned row count land exactly on the
   published 395,147. The `python/diagnose_post_cutoff.py` diagnostic shows those 5,606 rows
   are **not** echoes of the deployed estimator (median residual ~1.2 m from the published
   formula, matching genuine depth rows; ~1% within 5 cm) — they are genuine depth. An earlier
   revision of this caveat called them "usable as bonus data in a refit"; **they are not**:
   verified against production 2026-08-07, every one of them is a `tutorial` label (the
   tutorial runs on a fixed legacy pano, which is why depth kept serving it), so the standard
   tutorial filter removes them from any analysis population.
2. `tutorial` uses the 2021-faithful definition (the boolean flag; DC via
   `gsv_onboarding_pano` membership). Today's application additionally excludes labels by
   tutorial street edge, which the 2021 analysis did not.
3. The 2021 raw total (492,299) is not comparable to this export's raw total (468,608): the
   2021 CSVs included non-depth rows that the cleaning filters then removed. The comparable
   number is the post-cleaning count, which matches exactly.
4. Booleans are Postgres CSV `t`/`f`; both `readr` and the Python loader parse them.

## Regenerating

Run `scripts/extraction/extract-depth-labels.sh --dc-db sidewalk_dc` on the database host (see
script header), replace the `.csv.gz` files here, then re-run `scripts/rerun-analysis.R`
(refreshes `tests/fixtures/r-baseline/`), update the counts above, and run `pytest`.

---

# Depth-pilot artifacts (issue #4)

The `depth-pilot-*` files are the committed evidence of the 2026-08-05 streetlevel depth pilot
([issue #4](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/4), report:
`reports/2026-08-05-depth-pilot.md`). They were fetched from Google's **unofficial, keyless
photometa endpoint** via [sk-zk/streetlevel](https://github.com/sk-zk/streetlevel) 0.12.10 on
2026-08-05 (UTC) and are committed precisely because that endpoint can vanish — the raw payloads
make the pilot's entire analysis reproducible offline forever.

- `depth-pilot-payloads.jsonl.gz` — one line per panorama that served depth (409 unique): pano id,
  fetch timestamp, capture date, advertised image sizes, and the **verbatim base64 depth payload**
  (Google's plane-fit synthetic depth, 512×256; not imagery). Decode with
  `python/gsv_depth.py::decode_depth_payload`.
- `depth-pilot-panos.csv.gz` — one row per sampled panorama: Part A (606 ids drawn seed-666 from
  the recovered dataset; resolve/`gone` status, payload-vintage class, re-registration deltas) and
  Part B (200 current panos at recovered-label locations in seattle/cdmx); per-pano camera-height
  QC from the payload's plane list.
- `depth-pilot-labels.csv.gz` — one row per label on every Part A pano that served depth: the
  stored 2017–2020 depth-derived position vs the position recomputed from today's payload with the
  bit-exact 2020 client algorithm, deltas in float32 ulps and meters, and the edge-case flags
  (no-plane, seam-wrap, stored-absurd).
- `depth-pilot-summary.json` — the headline numbers, asserted by `tests/test_depth_pilot_findings.py`.

**Regenerate by intent only**: `python python/run_depth_pilot.py fetch` then `build` (then update
the findings tests and the report — a refetch observes a *different* GSV state, so drift in the
numbers is expected, not an error). `build` alone is deterministic: re-running it against the
gitignored `data/depth-pilot-cache/` reproduces these files byte-for-byte.

---

# Depth-validation artifacts (issue #9)

The `depth-validation-*` files are the committed evidence of the 2026-08-06 depth validation
([issue #9](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/9); reports:
`reports/2026-08-06-depth-validation.md` and `reports/2026-08-06-depth-coordinate-conventions.md`).
They exist so the conclusions can be re-derived rather than trusted: `fetch` is the only networked
stage, and `build` / `figures` / `gallery` / `verify_depth_conventions.py` all replay these bytes
with the cache deleted and no network at all.

- `depth-validation-tiles.jsonl.gz` — **the replication artifact.** One line per panorama-zoom
  record: pano id, set (`scoring` 60 panos / `adjudication` 24), zoom index, pixel dimensions, tile
  grid, and the **verbatim JPEG bytes of each tile as Google served them**, base64-encoded. Not a
  re-encode, so a refetch can be diffed against it and the stitch is deterministic
  (`depth_validation.stitch_tiles`). Same principle as `depth-pilot-payloads.jsonl.gz`: an overlay
  result nobody can regenerate is not evidence.
- `depth-validation-panometa.csv.gz` — per-panorama yaw, pitch, roll and road-link bearings for all
  409 panoramas that served depth. Small but load-bearing: the yaw is required to place a label on
  the raster at all, and the link bearings are the external reference that identifies which frame
  the raster is in.
- `depth-validation-panos.csv.gz` — per panorama: registration scores under each frame control, the
  permutation-null summary, the column-sweep minimum, plane inventory, flat-earth comparison, and
  the `structure_fraction` that says whether the panorama can testify about registration at all.
- `depth-validation-labels.csv.gz` — per label: which surface its pixel lands on, range, height
  above ground, the flat-earth counterfactual, and the curb-height bias estimate.
- `depth-validation-crossvintage.csv.gz` — 29 panoramas paired with a historical capture of the
  same location (median 11 years apart): ground-surface residuals and facade plane offsets.
- `depth-validation-partners.jsonl.gz` — the historical partners' depth payloads, verbatim.
- `depth-validation-sweeps.json.gz` — the per-panorama column-offset sweep curves.
- `depth-validation-adjudication.json` — the hand occlusion verdicts. A recorded human judgement,
  not a computation: occlusion is geometrically undetectable, because an unmodelled car returns
  exactly the range flat earth predicts. Carries a `_correction` field; an earlier pass was wrong
  for the coordinate-frame reason set out in the conventions report.
- `depth-validation-summary.json` — the headline numbers, asserted by
  `tests/test_depth_validation_findings.py`.
- `depth-conventions-evidence.json` — output of `python/verify_depth_conventions.py`, asserted by
  `tests/test_depth_conventions.py`.

**Regenerate by intent only**: `python python/run_depth_validation.py fetch` then `build`. A refetch
observes a different GSV state, so drift in the numbers is expected rather than an error; update the
findings tests and the reports alongside it.

---

# POV-inversion summary (issue #5)

`pov-inversion-summary.json` is the committed evidence of the 2026-08-06 exact-POV-inversion
comparison ([issue #5](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/5);
report: `reports/2026-08-06-pov-inversion.md`), asserted by
`tests/test_pov_inversion_findings.py`. Unlike the depth artifacts it needs no fetch at all —
it derives entirely from the `labels-*-latlng.csv.gz` files and the R-fixture split, so
`python python/run_pov_inversion.py --write` regenerates it deterministically on any machine,
and the findings tests re-derive its headline numbers in-process from the same code.

# Mapillary-falsification inputs (issue #3, Stage 3)

The `falsification-*` files are the Stage 3 inputs: the auto-labeler's fused multi-view
curb-ramp sites for the two Mapillary-viewer cities (**richmond**, **clovis**) and the four GSV
controls (**paterson**, **gainesville**, **bend**, **sao_paulo**), plus per-panorama metadata.
They are committed here because both are gitignored artifacts of the
[sidewalk-auto-labeler](https://github.com/ProjectSidewalk/sidewalk-auto-labeler) repo's run
directories — same preservation principle as the depth payloads: evidence nobody can regenerate
is not evidence.

- `falsification-sites-<run>.jsonl.gz` — `fuse_sites.py` output, verbatim: one fused site per
  line (GLS position + covariance) with per-member detections (`x/y_normalized`, ray
  `range_m`/`bearing_deg`, per-member lat/lng, capture date). All six runs were fused with
  identical default parameters (recorded in the meta JSON); clovis/gainesville/bend/sao_paulo
  were fused 2026-08-07 at auto-labeler `0bbd8e6`, and regeneration at that commit was verified
  byte-identical against the run-time richmond output.
- `falsification-panos-<run>.csv.gz` — one row per panorama in the run's `results.jsonl`:
  position, dimensions, pose, detection count, and (Mapillary) the full Graph API census
  fields — `camera_type`/make/model, `sequence_id`, raw vs `computed_*` geometry / compass /
  altitude, SfM `computed_rotation` and `atomic_scale`, creator, `quality_score`. GSV rows
  leave the Mapillary-only columns empty. Covers **all** panos the run saw (72,776 for clovis),
  not just those with detections, so the census is unconditioned on the detector.
- `falsification-runs-meta.json` — per-run fuse parameters/counts, height×width histograms,
  and SHA-256s of the source files, plus the auto-labeler commit.

Regenerate with `python python/import_falsification_inputs.py <auto-labeler>/runs`
(deterministic; byte-identical across reruns). The fused sites depend only on each run's
committed-there `results.jsonl`, so the import is reproducible from an auto-labeler checkout at
the recorded commit.

# Distance-refit summary (issue #3)

`distance-refit-summary.json` is the committed evidence of the 2026-08-07 distance-half refit
([issue #3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3), Stages 1–2;
report: `reports/2026-08-07-distance-refit.md`), asserted by
`tests/test_distance_refit_findings.py`. Like the POV-inversion summary it needs no fetch —
it derives from the `labels-*-latlng.csv.gz` files, the R-fixture split, and two committed #4/#9
artifacts (`depth-pilot-panos.csv.gz` for served camera heights, `depth-validation-panometa.csv.gz`
for the tilt rider) — so `python python/run_distance_refit.py --write` regenerates it
deterministically on any machine (byte-identical across reruns), and the findings tests re-derive
its headline numbers in-process from the same code. Contents: the full rung × loss results matrix,
`bounds` (each form's *structural* maximum — the largest distance it can return anywhere in the
depression domain, which is what the report's near-horizon claims mean, as opposed to the largest
one the thin near-horizon test slice happened to draw), the fixed-frame and #4765 apply-path
checks, near-horizon and click-noise robustness tables, per-type camera heights with the pooled
fallback for unseen label types, riders, and the era fit's coefficient hand-off
(`era_fit_coefficients` — final in *form*, but its height scale carries the era truth's
pinned-plane artifact; the calibrated production constants live in
`modern-truth-summary.json` `final_coefficients`).

Two test files stand behind it: `tests/test_distance_refit_findings.py` locks what this run
measured, and `tests/test_distance_refit_contract.py` locks what must hold for *any* refit —
solver exactness, boundedness and monotonicity of every form, the unseen-label-type fallback,
and the harness properties the ladder's comparisons rest on.

# Falsification and GBM-ceiling summaries (issue #3 Stage 3, issue #6)

`falsification-summary.json` (census + the two scale-free diagnostics + per-sequence camera
heights with the held-out seed sweep; report `reports/2026-08-07-mapillary-falsification.md`)
and `gbm-ceiling-summary.json` (the LightGBM benchmark matrix, ablation, importances and the
shared noise sweep; report `reports/2026-08-07-gbm-ceiling.md`) are the committed outputs of
`python python/run_mapillary_falsification.py --write` and `python python/run_gbm_ceiling.py
--write` respectively — both deterministic from committed inputs (byte-identical across
reruns), both locked by their `tests/test_*_findings.py`.

# Modern-truth validation set (issue #3, absolute-scale close-out)

The `modern-truth-*` files are the absolute check the Mapillary falsification could not do:
post-2021 human labels (whose stored `pano_x`/`pano_y` replay the front-end projection
exactly) scored against fresh GSV depth fetched by pano id on 2026-08-07. Report:
`reports/2026-08-07-modern-truth.md`.

- `modern-truth-payloads.jsonl.gz` — verbatim base64 depth payloads, one line per pano that
  resolved and served depth (1,106 unique; ~5 MB). Same preservation principle as the
  depth-pilot payloads: the evidence, not a pointer to it.
- `modern-truth-panos.csv.gz` — one row per **attempted** pano (1,911): fetch status
  (`ok`/`gone`/`parse_error`), stratum, fresh pose/position/capture date, camera-height QC
  (including the exactly-2.50 m pinned-plane flag), and the origin drift vs the extraction's
  `pano_data` position.
- `modern-truth-labels.csv.gz` — one row per frame-gated label on an ok pano (3,286): the
  extraction columns, the depth hit (class, ray, horizontal truth, neighbourhood ratio),
  truth gates, all four model predictions, and the era-aware circularity-guard fields.
  **Keyed by `label_uid` (`city:label_id`), never `label_id`** — that column is a
  per-schema serial and 76% of the extraction's rows share one with another city, so a join
  on it alone silently pairs a label with a different city's depth truth.
- `modern-truth-summary.json` — the findings `tests/test_modern_truth_findings.py` locks:
  fetch/gate censuses (including realized per-type label delivery against the quota, and how
  many cities cleared the by-city minimum), the two-era guard, the model matrix by stratum,
  near-horizon bins, implied per-type heights, frame-control sweep, the held-out remedy check
  with the deployed model scored on the same rows, and **`final_coefficients`** — the Stage 4
  production constants (the blend form with one flat 2.34 m height; decision and tradeoffs in
  the report's §9).

The sampling frame is the (uncommitted, regenerable) all-city extraction under
`modern-extraction/` — `scripts/extraction/extract-modern-labels.sh` rebuilds it read-only
from production; per-city population censuses live in its `extraction-metadata.txt`. The
fetch cache (`modern-truth-cache/`) is gitignored; `python python/run_modern_truth.py build
--write` replays offline from the cache (byte-identical across reruns), and the findings
tests re-derive the headline numbers from the committed payloads + labels — every truth
value on a 250-pano slice by default, and on **all** 3,286 rows under `RUN_SLOW=1`
(`pytest tests/test_modern_truth_findings.py`), so the artifacts and the code cannot drift
apart. **Regenerate by intent only**: a refetch observes a different GSV state (panos die,
depth planes get re-measured), so drift in a fresh fetch is expected rather than an error.

---

# Bearing-only triangulation artifacts (issue #7)

Written by `python python/run_triangulation.py`. The inputs are the **already-committed**
auto-labeler multi-view runs (`falsification-sites-*.jsonl.gz`, `falsification-panos-*.csv.gz`,
imported for issue #3 Stage 3) — no new extraction and no database. The only stage that
touches the network is `fetch`, and its payloads are committed verbatim, so `build`,
the figures and the tests all replay from a fresh checkout.

- `triangulation-summary.json` — the findings `tests/test_triangulation_findings.py` locks:
  per-run applicability and intersection-angle conditioning, the converged noise budget
  (σ_bearing / σ_panorama-position, with the iteration trace and the binned regression the
  split comes from), synthetic **and** real-geometry bias validation, the implied camera
  height by two estimators with site-bootstrap intervals, the robustness sweeps (conditioning
  gate, site size, fuse-gate selection probe, and the **rejected** camera-tilt hypothesis
  under all four sign conventions), every distance model scored against the triangulated
  truth, split-half precision, the fitted global bearing offset, cross-source absolute rig
  heights, and the same-pixel `depth_anchor` — including the position-drift check (stored
  vs freshly fetched panorama positions) and the gap's range and capture-era profiles,
  which separate the §8 candidates by shape.
- `triangulation-depth-payloads.jsonl.gz` — verbatim base64 depth payloads for 480 GSV
  panoramas (120 per GSV run; ~3 MB), fetched 2026-08-08. Same preservation principle as the
  depth-pilot and modern-truth payloads: the evidence, not a pointer to it.
- `triangulation-depth-panos.csv.gz` — one row per attempted pano (480, all `ok`): fetch
  status plus the fresh photometa pose, position and capture date. The position column is
  what proves the auto-labeler's stored panorama positions are Google's own (median drift
  0.000 m, computed as `depth_anchor.position_drift` and locked by the findings tests),
  which is load-bearing because triangulated range scales with the baseline.
- `triangulation-viz-depth-payloads.jsonl.gz` — verbatim depth payloads for the showcase
  cameras that fall **outside** the §8 anchor's 480-pano sample, so every camera view on
  the conclusions page carries its depth-model panel. Deliberately separate from
  `triangulation-depth-payloads.jsonl.gz`: the anchor population is locked by the
  findings tests and must not grow, and these pixels carry no committed `r_depth`.
- `triangulation-viz-tiles.jsonl.gz` — verbatim GSV imagery tiles (192 tiles, 18
  panoramas, ~7 MB, fetched 2026-08-09) behind `figures/triangulation-conclusions.html`,
  the interactive conclusions page built by `python/triangulation_viz.py`. Context for
  human eyes: no number in the reports or tests depends on these bytes — the page's
  charts replay `triangulation-summary.json` and its depth panels replay the committed
  payloads. Committed per the archival rule in `CLAUDE.md`: every byte an artifact
  depends on lives in the repo, so the page rebuilds offline from a fresh checkout.

**What the truth here does and does not depend on.** It uses panorama positions, panorama
headings and the horizontal detection angle. It uses **no** vertical click angle, camera
height, ground-plane assumption, depth data or panorama resolution — that is the entire
point, and `test_declared_inputs_exclude_every_vertical_and_depth_quantity` pins the claim.
Two properties are inherited rather than established here and are stated in the report:
cluster membership comes from the auto-labeler's fuse (which assumed a 2.6 m camera height;
the selection probe tests whether that biased the answer, and finds it did not), and the
whole chain still rests on the camera positions Google and Mapillary report — a weaker
dependency than trusting their depth planes, but not none.

Regenerating: `python python/run_triangulation.py build --write` (~12 min, offline). A
re-`fetch` observes a different GSV state and will drift; regenerate by intent only.
