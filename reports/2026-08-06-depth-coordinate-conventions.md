# GSV depth: coordinate conventions, and how streetlevel compares to the old technique

**2026-08-06** · issue [#9](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/9) ·
companion to [the depth validation report](2026-08-06-depth-validation.md)

> **Reproduce every number here in one command**, offline from committed bytes:
> ```bash
> python python/verify_depth_conventions.py --json   # -> data/depth-conventions-evidence.json
> pytest tests/test_depth_conventions.py             # the conclusions, locked
> RUN_SLOW=1 pytest tests/test_depth_conventions.py  # ...and re-derived in-process
> ```
> Seven checks, labelled A–G below and in the script's output. No network; the only
> decode path is the v6 replica in `python/gsv_depth.py`.

Written because Project Sidewalk is wiring streetlevel depth into the panorama downloader
([sidewalk-panorama-tools#39](https://github.com/ProjectSidewalk/sidewalk-panorama-tools/pull/39)),
and three separate coordinate conventions are in play that do not agree with each other. Getting
any of them wrong is silent: the arrays are the right shape, the numbers are plausible metres, and
nothing raises.

## §1 · Three frames, and which is which  ·  checks B, C, D, E

| thing | frame | how to know |
|---|---|---|
| `sv_image_x` (pre-evolution-179, drives the depth lookup) | **north-referenced**: `sv_image_x / 13312 × 360` is the label's true compass bearing | over all 395,147 cleaned labels, this minus the independently recorded POV `heading` is centred on −0.3° with 99.99% inside ±60°; the heading-shifted alternative keeps only 32% (std 105°) |
| `pano_x` / `current_pano_x` (post-evolution-179, drives cropping) | **heading-centred** | differs from `sv_image_x/13312` by exactly `(180 − pano_yaw)/360`, r = 0.954 over 195,556 rows |
| the panorama raster, and the depth payload | **heading-centred**: column 0 is bearing `pano_yaw − 180`, so the vehicle's forward direction sits at image centre | Project Sidewalk's own 2017 `GSVImage.py`: `heading = 360·(x/width) + (pano_yaw_deg − 180)`. Independently: road links land on road (0.94 vs 0.87 road-likeness on 85 discriminating links) and depth sightlines run long down the street (84% vs 66% of 655 links) |

So `sv_image_x` and the raster are **not** the same coordinate. Converting a label to a pixel:

```python
bearing = sv_image_x / 13312 * 360                      # true compass bearing
col     = ((bearing - pano_yaw_deg + 180) % 360) / 360  # fraction of image width
row     = (3328 - sv_image_y) / 26 / 256                # fraction of image height
```

The rotation is up to half a panorama and vanishes only for panoramas that happen to face due
south — which is exactly why the error hides. It was caught by eye on a DC panorama
(`8t4iLIsgG5N4j3gN3wvKSg`, yaw 101.5°) whose labels sat in a planting bed; with the rotation
applied all four land on curb ramps and a grass verge. A Columbus panorama at yaw 171.9° looked
almost correct, because there the two conventions differ by 8°.

Vertical needs no rotation: `sv_image_y` is measured from the horizon, which is payload row 128 and
image row `height/2`.

**Open item (check G).** The 2017–2020 client indexed the depth payload with `ceil(sv_image_x/26)` — a
north-referenced coordinate against a heading-centred map. If the frames are as established above,
the stored *bearing* is right (the client derived it from the same coordinate) but the stored
*distance* was sampled from a column `(180 − pano_yaw)/360 × 512` away. Re-reading every label
(the 837 pilot labels that survive cleaning) at the heading-centred column moves the median
distance by **0.00 m** and 85% by under 1 m — because
the model is nearly flat earth, so range is set by the depression angle and barely by azimuth — but
**6.6% move by more than 3 m** (p95 = 4.1 m). This wants a second reviewer before it is treated as
settled; it bears directly on the ground truth behind
[#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3).

## §2 · streetlevel's depth array is x-mirrored  ·  checks A, F

`streetlevel.streetview.depth.parse(...).data` writes the value computed for payload column *x*
at output column *w − 1 − x* (`depth.py`: `depth_map[y * w + (w - x - 1)] = t`). The 2017–2020
Project Sidewalk client did not mirror; nor does `python/gsv_depth.py`, which replicates it.

Checked over all 409 committed payloads: unmirrored they differ by a median of **75 m** (worst
882 m); mirrored, **all 409 agree to within 10 cm** — median 15 µm, and the handful above 1 mm is
float32 quantization in the v6 replica at long range, not a disagreement about geometry. So the two
are the same algorithm differing only by an x-mirror; there is no content difference at all.

Because the payload shares the raster's frame, the mirrored array is the exact condition this
repo's registration test rejects: `x_mirror` loses to the true frame on 41 of 53 panoramas, and
loses outright on both vertical flips 52–1.

To index streetlevel's array with a heading-centred `pano_x`:

```python
col_depth = 511 - int(round(pano_x / pano_width * 512))   # note the mirror
row_depth = int(round(pano_y / pano_height * 256))        # row 128 is the horizon
```

## §3 · Is it the same depth map the old technique fetched?

**Yes — same product, different wrapper.**

| | old (2017 – ~2020) | now |
|---|---|---|
| transport | `maps.google.com/cbk?output=xml&dm=1` → base64 in the XML | photometa (`maps.googleapis.com`) → base64 in the JSON |
| compression | zlib-deflated | raw |
| decoder | `decode_depthmap`, a compiled binary committed to sidewalk-panorama-tools in 2017 | `streetlevel`, or `gsv_depth.decode_depth_payload` |
| structure | 8-byte header, plane-index raster, plane list | identical |
| geometry | 512×256, angular; row 128 = horizon | identical |

`gsv_depth.decode_depth_payload` accepts both and records which it saw in `was_compressed`, so the
two are directly comparable. The #4 pilot measured the *content* agreement against label positions
computed by the old pipeline in 2017–2020: **23% of surviving panoramas are bit-stable** (agreement
at the float32 storage floor) and the rest drift under Google's reprocessing, 0.98 m median overall.
So it is the same synthetic product, re-served; it is not frozen, and older captures have drifted
more.

## §4 · What the product is, in one paragraph

Not a measurement. It is a constructed model of terrain plus extruded building footprints: 84.5%
of pixels sit on near-flat surfaces (within 10° of horizontal) and 13.7% on near-vertical ones
(within 10° of a wall), with only **0.85%** anywhere between 15° and 75° — no car roofs, no tree
canopy, no pitched roofs, no driveway ramps. Under a Sidewalk label, the median panorama has 91%
of its ground-band pixels within 1 m of naive `h/tan(depression)`.
Consequences for anything consuming it: vehicles, people and vegetation are absent, so a ray aimed
at one passes through and returns the ground behind (a distance *overestimate*, undetectable
geometrically); a curb ramp sits ~0.15 m above the modelled road, so rays overshoot by ~0.5 m at
typical label distances; and per-panorama camera height (median 2.37 m) is recoverable from the
payload's plane list, but only if the plane list is kept. Full evidence:
[the depth validation report](2026-08-06-depth-validation.md).

---

*Written with [Claude Code](https://claude.com/claude-code) (claude-opus-5[1m]). The frame error in
§1 was found by Jon spotting that gallery labels sat left of their features.*
