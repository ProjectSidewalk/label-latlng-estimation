# The html4-era DC rows: a quarter of DC's ground truth was clicked on a flat renderer

**Date:** 2026-11-05. **Evidence:** `data/html4-era-summary.json`, regenerated offline by
`python python/run_html4_era.py --write` from the committed `labels-dc-latlng.csv.gz`; locked by
`tests/test_html4_era_findings.py`. Every number below is in that JSON unless it is marked as coming
from somewhere else. **Upstream:** [SidewalkWebpage#5920](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5920)
(repair PR [#5923](https://github.com/ProjectSidewalk/SidewalkWebpage/pull/5923)).

## What was found upstream

Until Google retired Maps JS API 3.25 on 2017-02-15, the Project Sidewalk client requested
`mode: 'html4'`: a flat, non-perspective Street View renderer. The client recorded `sv_image_x/y`
with a matching **linear** formula on a fixed 13312×6656 frame (4.6 frame px per canvas px in x,
4.65 in y, divided by a zoom factor of 1 / 2.1 / 4), and kept that formula until v2.1 on
2017-03-21. Every reader written since, including this repo's `pov_if_centered`, projects the
stored heading/pitch/zoom/canvas record with perspective math. On a flat-renderer click that is
the wrong forward model. DC was the only city before 2017, so only DC has such rows.
SidewalkWebpage#5920 re-solves the stored heading/pitch of those labels in production so that the
perspective readers hit the original click. The legacy `sidewalk_dc` database this repo's extract
came from is not touched by it.

In this repo's DC extract the rows are `label_id < 66003` (the first row the v2.1 client wrote;
`html4_era.is_html4`). Their `time_created` is NULL here, so a date filter cannot find them; the
id can. Upstream, the repair cuts at the renderer switch on 2017-02-15 (label 65960) rather than
the formula switch, because the 13 labels in between were recorded with the linear formula but
seen through perspective (SidewalkWebpage#5920, the cut section of the script); too few to matter
here.

## What the committed data shows

| | html4 rows (label_id < 66003) | later DC rows |
|---|---|---|
| rows raw / cleaned | 65,815 / 56,347 | — / 173,986 |
| stored `sv_image_x` within 1 px of the linear replay (`linear_sv_image`) | 99.99% | 0.5% |
| stored `sv_image_x` within 1 px of the era client's perspective replica (`era_sv_image_x`: `parseInt`-truncated inputs and a half-pixel offset, the POV-inversion report §3) | 0.4% | 99.8% |
| per-row classifier `stored_formula` | 56,093 linear, 250 both (centre clicks), 4 neither, 0 era | 172,843 era, 824 both, 268 neither, 51 linear |
| median \|bearing residual\| of the stored lat/lng, perspective click bearing | **3.08°** | 1.10° |
| median \|bearing residual\|, linear click bearing | 0.94° | 3.04° |
| median signed bearing residual under the row's own formula | +0.60° | +0.74° |
| implied camera height (`pano_dist · tan depression`, depression ≥ 5°), perspective | 2.45 m | 2.85 m |
| implied camera height, linear | 2.77 m | 3.14 m |

The 0.9–1.1° absolute floor that remains under the right formula in each column, and the +0.6 to
+0.7° signed median, are the depth-grid `Math.ceil` bias the POV-inversion report measured at
about +0.72°, plus click noise.

**How far apart the two models are, per row** (`model_gap_deg`, html4 rows): in bearing 3.16° at
the median, 5.10° at p90, 8.71° at p99, 18.5° at most; in altitude 1.88° median, 3.85° p90,
5.89° p99, 11.8° at most. The gap is not a function of distance from the centre alone: the linear
slope was fitted to the whole canvas width, so the two models agree at the centre and come close
again at the edges, and differ most about 200 px out. The signed bearing residual against the
perspective bearing at zoom 1 runs +3.7, +5.0, +4.5, +2.2, −0.9, −3.2, −3.7, −2.4° across the
eight 90-px canvas-x bins, and the linear-minus-perspective altitude runs −4.0, −3.5, −2.0, −0.1,
+1.3, +2.6° down the six 80-px canvas-y bins. Later DC rows show a flat +0.7–0.8° in bearing.
The large tail comes from steep pitches, where the perspective inversion couples the two axes.

The html4 rows are 24% of cleaned DC. The production sign-off report (§4.3) puts DC at 58% of the
era test split, so roughly 14% of the era corpus is html4 (`clean_data` keeps NULL `time_created`,
so they are in every era split).

## How to read the reports that pool them

The stored lat/lng of an html4 row was computed by the 2017–2020 client from its *linear*
`sv_image_x/y`, so as a truth target it is consistent with the click that produced it. What is
wrong on those rows is every estimator **input** this repo derives from the POV record through
`pov_if_centered`: `exact_heading_diff`, `exact_depression_deg`, `era_heading_diff`, and anything
built on them. Measured against that truth, those inputs are off by the table above. The stored
`sv_image_y` itself is also a different function of (`canvas_y`, `pitch`) on these rows than on
the rest of the era corpus.

- **POV inversion (2026-08-06).** §2's pooled "sv-implied" zoom-1 slope (0.1443°/px, read as the
  era client's `parseInt` signature) mixes the html4 rows' 0.1244°/px linear slope (4.6 · 360 /
  13312) with the later rows' truncated perspective. §4's "no canvas structure in the bearing
  bias (R² ≤ 0.006)" was measured on a pool that carries a ±5° S-curve in canvas x on a quarter of
  DC; the constant +0.72° stands (signed medians +0.60° and +0.74° above). The medians and the
  zoom 2–3 comparison include the html4 rows.
- **Distance refit (2026-08-07).** Its vertical input is `exact_depression_deg` for all 395k era
  rows, html4 included, so the era fits and their camera-height constants carry the altitude gap
  above.
- **GBM ceiling (2026-08-07) and GBM transfer (2026-08-10).** The headline boosters take the
  stored `sv_image_y` (`run_gbm_ceiling.FEATURES_FULL`), not the perspective depression; only the
  `gbm_dep_l1` and `only_depression` variants take `depression_deg`. The exposure there is that
  `sv_image_y` means a different function of the record on 14% of the training rows, which a
  booster given `heading`, `pitch`, `canvas_y` and `pano_height` can learn to tell apart. The
  statement that the era truth "implies 2.80 m of camera height at DC" is a blend: later DC rows
  imply 2.85 m under perspective, html4 rows 2.45 m under perspective but 2.77 m under their own
  formula. The pinned 2.50 m payload planes remain the main explanation; the html4 rows widen
  the DC scatter rather than move its centre much.
- **Production sign-off (2026-09-02).** The era-frame tables (§4.3) and the DC slice in §5.1 score
  `approximation3` on inputs that are wrong on a quarter of DC. The modern-truth results, the
  leave-one-city-out calibration, and `final_coefficients` do not include DC and are untouched.

Nothing shipped changes: `final_coefficients` were calibrated on modern truth, which has no DC
rows, and every headline in this repo is scored in that frame. The era-frame numbers above would
tighten with the html4 rows excluded (`html4_era.is_html4`) or replayed through
`html4_era.linear_sv_image`; no conclusion is expected to flip, so the reports are left as written
with this note as the caveat. A dispatch inside `pov_inversion` that routed html4 rows through the
linear formula would fix every caller at once, but it would move every locked era-frame finding
and so is a change of its own, not part of this note.

If DC's era truth is ever re-extracted from production after the #5920 repair lands, the stored
heading/pitch of these rows will have changed so that `pov_if_centered` reproduces the original
click: `stored_formula` will then call them `era`, `linear_sv_image` must not be applied to them,
and `is_html4` (by id) stays the marker. The stored `sv_image_x/y` and lat/lng do not change.
