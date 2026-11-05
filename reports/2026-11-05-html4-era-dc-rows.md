# The html4-era DC rows: a quarter of DC's ground truth was clicked on a flat renderer

**Date:** 2026-11-05. **Evidence:** `data/html4-era-summary.json`, regenerated offline by
`python python/run_html4_era.py --write` from the committed `labels-dc-latlng.csv.gz`; locked by
`tests/test_html4_era_findings.py`. **Upstream:** [SidewalkWebpage#5920](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/5920)
(repair PR [#5923](https://github.com/ProjectSidewalk/SidewalkWebpage/pull/5923)).

## What was found upstream

Until Google retired Maps JS API 3.25 on 2017-02-15, the Project Sidewalk client requested
`mode: 'html4'`: a flat, non-perspective Street View renderer. The client recorded `sv_image_x/y`
with a matching **linear** formula on a fixed 13312×6656 frame (4.6 frame px per canvas px in x,
4.65 in y, divided by a zoom factor of 1 / 2.1 / 4), and kept that formula until v2.1 on
2017-03-21. Every reader written since, including this repo's `pov_if_centered`, projects the
stored heading/pitch/zoom/canvas record with perspective math. On a flat-renderer click that is
the wrong forward model: the marker lands up to about 5° from the click, growing with distance
from the canvas centre (top clicks too high, bottom too low, side clicks too far out). DC was the
only city before 2017, so only DC has such rows. SidewalkWebpage#5920 re-solves the stored
heading/pitch of those labels so that the perspective readers hit the original click.

In this repo's DC extract the rows are `label_id < 66003` (the first row the v2.1 client wrote).
Their `time_created` is NULL here, so a date filter cannot find them; the id can. The repair
upstream cuts at the renderer switch (label 65960, 2017-02-15 17:00 UTC) rather than the formula
switch; the 13 rows in between were recorded with the linear formula but seen through perspective,
which is too few to matter here.

## What the committed data shows

| | html4 rows (label_id < 66003) | later DC rows |
|---|---|---|
| rows raw / cleaned | 65,815 / 56,347 | — / 173,986 |
| stored `sv_image_x/y` within 1 px of the linear replay | 99.99% | 0.04% |
| stored `sv_image_x/y` within 1 px of `pov_if_centered` | 0.01% | 1.1% (99.8% with the era replica's `parseInt` and half-pixel; see the POV-inversion report §3) |
| median \|bearing residual\| of the stored lat/lng, perspective click bearing | **3.08°** | 1.10° |
| median \|bearing residual\|, linear click bearing | 0.94° | 3.04° |
| implied camera height (`pano_dist · tan depression`, depression ≥ 5°), perspective | 2.45 m | 2.85 m |
| implied camera height, linear | 2.77 m | 3.14 m |

The 0.9–1.1° floor that remains under the right formula in each column is the depth-grid `Math.ceil`
bias (about 0.72°) plus click noise, as the POV-inversion report measured.

The error has the projection signature. On html4 rows at zoom 1, the signed bearing residual
against the perspective bearing runs +3.7, +5.0, +4.5, +2.2, −0.9, −3.2, −3.7, −2.4° across the
eight 90-px canvas-x bins, and the linear-minus-perspective altitude runs −4.0, −3.5, −2.0, −0.1,
+1.3, +2.6° down the six 80-px canvas-y bins. Later DC rows show a flat +0.7–0.8° in bearing.

The html4 rows are 24% of cleaned DC and, with DC at 58% of the era test split, roughly 14% of the
era corpus (`clean_data` keeps NULL `time_created`, so they are in every era split).

## How to read the reports that pool them

The stored lat/lng of an html4 row was computed by the 2017–2020 client from its *linear*
`sv_image_x/y`, so as a truth target it is consistent with the click that produced it. What is
wrong on those rows is every estimator **input** this repo derives from the POV record through
`pov_if_centered`: `exact_heading_diff`, `exact_depression_deg`, and anything built on them.
Measured against that truth, those inputs are off by the table above.

- **POV inversion (2026-08-06).** §2's pooled "sv-implied" zoom-1 slope (0.1443°/px, read as the
  era client's `parseInt` signature) mixes the html4 rows' 0.1244°/px linear slope with the later
  rows' truncated perspective. §4's "no canvas structure in the bearing bias (R² ≤ 0.006)" was
  measured on a pool that carries a ±5° S-curve in canvas x on a quarter of DC; the constant +0.72°
  stands (0.68° on html4 rows, 0.71° later). The medians and the zoom 2–3 comparison include the
  html4 rows.
- **Distance refit (2026-08-07), GBM ceiling (2026-08-07), GBM transfer (2026-08-10).** The era
  fits take `exact_depression_deg` as the vertical input for all 395k rows, html4 included. The
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
with this note as the caveat.

If DC's era truth is ever re-extracted from production after the #5920 repair lands, the stored
heading/pitch of these rows will have changed so that `pov_if_centered` reproduces the original
click; the stored `sv_image_x/y` and lat/lng do not change. `is_html4` keeps identifying the rows
by id either way.
