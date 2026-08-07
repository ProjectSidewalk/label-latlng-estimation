# Exact POV inversion: the heading half of the estimator is geometry, not statistics

**2026-08-06** · issue [#5](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/5) ·
feeds the refit in [#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3)

> **Reproduce every number here in one command**, offline from committed bytes:
> ```bash
> python python/run_pov_inversion.py --write     # -> data/pov-inversion-summary.json
> python python/pov_inversion_figures.py         # -> figures/fig13-pov-inversion.png
> pytest tests/test_pov_inversion_findings.py    # the conclusions, locked
> ```
> No network; no fitted parameters in the geometry. The math is
> `pov_if_centered` in `python/pov_inversion.py`.

est7's heading half is three per-zoom linear fits, `heading_diff ~ canvas_x` — six fitted
coefficients. Issue #5 conjectured they are a first-order approximation of the front end's own
projection geometry and could be replaced by running that geometry exactly. That is confirmed,
and chasing the one place the regression seemed to win exposed two measurable artifacts *in the
ground truth itself* that matter beyond this issue.

## §1 · The math, and the proof it is the production math

`pov_if_centered(canvas_x, canvas_y, heading, pitch, zoom)` replicates
`calculatePovIfCentered` / `calculatePointPov` from SidewalkWebpage's
`UtilitiesPanomarker.js` (verified identical at tags 6.13.0 and v7.19.10, and inlined verbatim
in evolution 179's SQL): a rectilinear camera at the POV, focal length
`f = 360 / tan(fov/2)` on the 720×480 canvas (every recovered row is 720×480), FOV per zoom
89.75° / 53° / 27.68° (`get3dFov`). It returns the click's absolute heading **and** its
elevation — the vertical output exists for #3's cotangent distance candidates
(`exact_depression_deg`).

Three independent fidelity checks against what production actually stored:

- **`pano_y`: 100.0000%.** Evolution 179 recomputed every `pano_y` from this projection;
  `replay_evolution_179` reproduces the stored integer exactly for **all** rows with pano
  metadata, in every city (e.g. 118,077/118,077 in Seattle). The vertical half has no free
  inputs, so this verifies the projection outright.
- **`pano_x` for post-cutoff labels: 100.0000% in all six cities.** Labels placed after
  evolution 179 got `pano_x` written live by the front end running this same math.
- **`pano_x` for pre-cutoff labels: 94.1% (Seattle) down to 46.9% (SPGG)** — and the misses
  are not projection error. `pano_x` alone consumes `camera_heading`, whose 2022 value (when
  the SQL ran) is unrecoverable where Google's pano metadata has since drifted. The signature
  is dispositive: among mismatching rows, the implied camera-heading delta is constant within
  a pano (median within-pano σ = **0.008°**, i.e. rounding noise) while varying across panos
  (σ = 0.73°). A per-pano metadata delta, not a math error.

## §2 · Scored against 2017–2020 targets: the regression was fitting artifacts

Same harness as #3 (R-fixture split, est7's fitted distance under every variant, so
differences are the heading half alone). Test set n = 79,029:

| heading model | fitted params | median heading err | median lat/lng err |
|---|---|---|---|
| est7 per-zoom linear fits | 6 | 1.3184° | 1.4621 m |
| exact inversion | 0 | 1.2500° | 1.4565 m |
| era-faithful exact (§3) | 0 | 1.2030° | 1.4546 m |
| era-faithful + one constant (§4) | 1 | **1.0306°** | **1.4469 m** |

The plain inversion wins pooled, and wins where #5 predicted the linear fit loses: the
outermost canvas bin (est7 **2.52°** vs exact **1.20°**) and the highest-|pitch| bin (1.72°
vs 1.41°). But est7 beat the plain inversion at zooms 2–3 (0.90° vs 1.06°, 0.70° vs 0.90°) —
the issue's "or something interesting is going on" branch. Two artifacts explain all of it,
and the fitted slopes had already told the story: est7's slopes match the *stored
`sv_image_x` mapping* to four decimals at every zoom (0.14434/0.07845/0.03963 fitted vs
0.14437/0.07839/0.03953 sv-implied) while the true projection's slopes differ
(0.14805/0.07955/0.03979). The regression was reproducing the legacy placement pipeline, not
the geometry.

## §3 · Artifact one: the era client truncated its POV inputs

The 2017–2020 `calculatePointPov` ran `parseInt(pov.heading)` and `parseInt(pov.pitch)` —
truncation to whole degrees — and `calculateImageCoordinateFromPointPov` added half of one
degree's width in pixels (+0.5°, ≈18.5 px). Replaying placement with those quirks reproduces
the stored `sv_image_x` for
**99.83%** of Seattle's pre-cutoff depth rows within 1 px (median 0.51 px, the int-storage
floor). Without the truncation: **8.0%**, median 16.8 px. The stored coordinates — and
therefore the lat/lng targets derived from them — carry sub-degree quantization noise that no
estimator can fit and every estimator pays for.

## §4 · Artifact two: the targets sit one depth-grid column clockwise

After the era-faithful model, the residual is a **constant**: +0.7232° / +0.7154° / +0.7007°
across zooms 1/2/3, 0.68–0.74° across all seven cities, with no canvas, pitch, or
bearing-harmonic structure (R² ≤ 0.006). One depth-grid column is 360/512 = **0.7031°**; the
train-set mean is **+0.7198°**.

The mechanism is in the era `Label.js` `toLatLng`:
`idx = 3 * (Math.ceil(p.x) + 512 * Math.ceil(p.y))` — the depth lookup **ceils** to the next
grid column (mean +½ column) and the payload's column azimuth convention supplies the other
half. This is the bearing-side twin of the
[conventions report's](2026-08-06-depth-coordinate-conventions.md) open item G (the same line's
distance-side column rotation). est7's per-zoom intercepts were absorbing it; with one shared
constant the geometry beats the six-parameter fit **at every zoom, every canvas position, and
every pitch band** (zoom 1: 1.16° vs 1.59°; zoom 2: 0.83° vs 0.90°; zoom 3: 0.65° vs 0.70°).

Crucially the constant is a property of the **2017–2020 ground truth**, not of clicks: it
exists because the legacy depth lookup biased the stored lat/lng, and it must **not** be
applied to anything produced after evolution 179.

## §5 · photographer_pitch: no heading signal

The scope note on #5 asked whether rig tilt leaks into the heading residual: it does not —
Pearson r = **−0.004**, slope −0.004°/° over a p5–p95 span of ±3.8°. (The *vertical* rig-tilt
question lives with the distance half and
[sidewalk-panorama-tools#54](https://github.com/ProjectSidewalk/sidewalk-panorama-tools/issues/54).)

## §6 · What this hands to #3 (and #7)

1. **The heading half is closed.** Production-facing form: `heading = pov_if_centered(...)`
   exactly — zero coefficients, valid at every zoom and canvas position, replacing six fitted
   numbers. `exact_depression_deg` supplies the vertical for #3's cotangent candidates from
   the same call.
2. **Score-time correction for legacy targets.** Any candidate scored against the 2017–2020
   ground truth should model the target bias: era-faithful inputs (§3) plus the one-column
   constant (§4), or it will be penalized ~0.7° for being right. #3's fitting harness gets
   both from `pov_inversion.era_heading_diff`.
3. **The ground truth's bearings are biased by +0.72°** (≈ 19 cm at the median 15 m label
   distance). That is invisible to est7-style refits (intercepts eat it) but matters for #7's
   triangulation — two biased bearings from different panos do not cancel — and for any
   depth re-lookup keyed by stored coordinates (conventions report, item G).

---

*Written with [Claude Code](https://claude.com/claude-code) (claude-fable-5). The zoom 2–3
anomaly that led to §3–§4 was the issue's own "if the exact inversion wins (it should, or
something interesting is going on)" — something interesting was going on.*
