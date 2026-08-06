"""Is Google's GSV depth authentic, and is it good? The analysis behind issue #9.

The issue #4 pilot compared fresh depth payloads against the stored 2017-2020 label
positions and found ~1 m agreement -- but both sides come from the same Google depth
product, so that validated *transport and stability* and said nothing about accuracy.
This module asks the two questions it left open:

1. **Authentic?** Does the payload describe THIS panorama's real scene? Settled by
   registering the depth against the panorama's own imagery -- which arrives from a
   *different Google host* (``streetviewpixels-pa.googleapis.com``) than the depth
   (``maps.googleapis.com`` photometa). Two independent endpoints cannot agree by
   decoding accident. The claimed mapping is pure scaling with no rotation offset,
   because ``13312 = 512 x 26``: depth column c <-> equirect column 26c at the
   13312-wide zoom, i.e. normalized column c/512 at any width. ``registration_scores``
   tests that claim against deliberately-wrong frames rather than assuming it.

2. **Good?** ``plane_inventory`` / ``tilt_histogram`` / ``flat_earth_comparison``
   characterize what the product actually IS. The answer is not a measurement: the
   plane set is a Manhattan world (essentially nothing tilted between 15 and 75
   degrees) and the ground is within a metre of naive ``h/tan(depression)`` almost
   everywhere. That matches sk-zk/streetlevel's README note -- "appears to be a
   synthetic depth map created from elevation data and building footprints" -- and
   turns it from a guess into a measurement.

Everything here is offline and deterministic: payload bytes come from
``data/depth-pilot-payloads.jsonl.gz`` and imagery from
``data/depth-validation-tiles.jsonl.gz``. Decoding is never reimplemented; the v6
replica in ``gsv_depth.py`` is the single decode path.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np

import gsv_depth as gd

# ---------- plane taxonomy
#
# Tilt is the angle of a plane's normal off vertical: 0 deg is a horizontal surface
# (ground, terrain, a flat roof), 90 deg is a wall. The oblique band is the one that
# matters -- a photogrammetric reconstruction of a street puts car roofs, tree canopy,
# pitched roofs, driveway ramps and hillsides there, and this product does not.
HORIZONTAL_TILT_DEG = 10.0
VERTICAL_TILT_DEG = 80.0
OBLIQUE_BAND_DEG = (15.0, 75.0)

# Depression band for the flat-earth comparison. Below 6 deg the ray runs to the far
# field where facades dominate; past 45 deg it approaches the nadir seam and the
# vehicle's own hood. At h = 2.4 m this band is roughly 2.4-23 m out, which brackets
# the 7-12 m where Sidewalk labels actually sit.
FLAT_EARTH_BAND_DEG = (6.0, 45.0)
FLAT_EARTH_MAX_RANGE_M = 80.0


def plane_tilt_deg(payload: gd.DepthPayload) -> np.ndarray:
    """Angle of each plane's normal off vertical, in degrees. 0 = flat, 90 = wall.

    Entry 0 is the no-plane marker, not a plane. Its stored normal is all zeros, which
    would come back as tilt 90 and read as a wall, so callers must mask index 0 out --
    use ``facade_pixel_mask`` rather than indexing this array by a raw pixel's plane id.
    """
    nz = np.abs(payload.planes_n[:, 2].astype(np.float64)).clip(0.0, 1.0)
    return np.degrees(np.arccos(nz))


def facade_pixel_mask(payload: gd.DepthPayload) -> np.ndarray:
    """Per-pixel boolean: does this pixel sit on a near-vertical plane (a wall)?

    Excludes the no-plane marker explicitly. Without that, every sky pixel reads as a
    wall, because plane 0's zero normal has an undefined tilt that ``arccos`` resolves
    to exactly 90 degrees.
    """
    idx = payload.indices.reshape(payload.height, payload.width)
    vertical = plane_tilt_deg(payload) >= VERTICAL_TILT_DEG
    vertical[0] = False
    return vertical[idx]


def plane_pixel_counts(payload: gd.DepthPayload) -> np.ndarray:
    """Pixels assigned to each plane. Index 0 (the no-plane marker) is zeroed."""
    counts = np.bincount(payload.indices, minlength=payload.n_planes).astype(np.float64)
    counts[0] = 0.0
    return counts


def ground_plane_index(payload: gd.DepthPayload) -> int:
    """The dominant near-horizontal plane below the horizon, or -1.

    Same rule as ``gsv_depth.camera_height_qc`` so the two never disagree about which
    plane is "the ground": most below-horizon pixels among near-vertical normals.
    """
    h, w = payload.height, payload.width
    below = payload.indices.reshape(h, w)[h // 2 :, :]
    counts = np.bincount(below.reshape(-1), minlength=payload.n_planes).astype(np.float64)
    counts[0] = 0.0
    vertical = np.abs(payload.planes_n[:, 2].astype(np.float64)) > 0.95
    if not vertical.any() or counts[vertical].sum() <= 0:
        return -1
    return int(np.where(vertical, counts, 0.0).argmax())


# ---------- T2: what the product actually is

@dataclass
class PlaneInventory:
    n_planes: int
    n_used: int
    n_horizontal: int
    n_vertical: int
    n_oblique: int
    px_share_horizontal: float
    px_share_vertical: float
    px_share_oblique: float
    px_share_no_plane: float
    horizontal_d_spread_m: float  # relief across the near-horizontal planes in use


def plane_inventory(payload: gd.DepthPayload) -> PlaneInventory:
    """Census of one payload's plane set, weighted by the pixels actually using it."""
    counts = plane_pixel_counts(payload)
    tilt = plane_tilt_deg(payload)
    used = counts > 0
    total_px = float(payload.indices.size)
    assigned = counts.sum()

    horiz = tilt <= HORIZONTAL_TILT_DEG
    vert = tilt >= VERTICAL_TILT_DEG
    obl = (tilt > OBLIQUE_BAND_DEG[0]) & (tilt < OBLIQUE_BAND_DEG[1])

    d = payload.planes_d.astype(np.float64)
    hd = d[horiz & used]
    spread = float(np.ptp(hd)) if hd.size > 1 else 0.0

    share = (lambda m: float(counts[m].sum() / assigned) if assigned > 0 else float("nan"))
    return PlaneInventory(
        n_planes=int(payload.n_planes),
        n_used=int(used.sum()),
        n_horizontal=int((horiz & used).sum()),
        n_vertical=int((vert & used).sum()),
        n_oblique=int((obl & used).sum()),
        px_share_horizontal=share(horiz),
        px_share_vertical=share(vert),
        px_share_oblique=share(obl),
        px_share_no_plane=float((payload.indices == 0).sum() / total_px),
        horizontal_d_spread_m=spread,
    )


def tilt_histogram(payload: gd.DepthPayload, bin_deg: float = 1.0) -> np.ndarray:
    """Pixel-weighted histogram of plane tilt over [0, 90) degrees.

    Counts, not fractions, so histograms from many payloads sum directly.
    """
    nbins = int(round(90.0 / bin_deg))
    hist, _ = np.histogram(
        plane_tilt_deg(payload), bins=nbins, range=(0.0, 90.0),
        weights=plane_pixel_counts(payload),
    )
    return hist


def sky_profile(payload: gd.DepthPayload) -> np.ndarray:
    """Fraction of no-plane pixels per payload row (row 0 = zenith, h/2 = horizon).

    The signature to look for is a step, not a slope: sky is unmodelled so it carries
    no plane, and the model's ground starts exactly at the horizon row.
    """
    idx = payload.indices.reshape(payload.height, payload.width)
    return (idx == 0).mean(axis=1)


def depression_angles(payload: gd.DepthPayload) -> np.ndarray:
    """Depression below the horizon per payload row, radians; negative above it.

    Mirrors ``compute_point_cloud``: phi is the polar angle from nadir, so
    depression = pi/2 - phi is positive for rows in the lower hemisphere.
    """
    y = np.arange(payload.height, dtype=np.float64)
    phi = (payload.height - y - 0.5) / payload.height * np.pi
    return np.pi / 2.0 - phi


@dataclass
class FlatEarthComparison:
    n_pixels: int
    median_residual_m: float  # observed horizontal range minus h/tan(depression)
    p10_residual_m: float
    p90_residual_m: float
    frac_within_1m: float
    frac_within_2m: float
    # where the metre-plus departures come from
    frac_dev_facade: float  # pixel is on a near-vertical plane: the ray hit a wall
    frac_dev_terrain: float  # a different near-horizontal plane: relief
    frac_dev_other: float


def flat_earth_comparison(
    payload: gd.DepthPayload,
    camera_height: float,
    band_deg: tuple[float, float] = FLAT_EARTH_BAND_DEG,
    max_range_m: float = FLAT_EARTH_MAX_RANGE_M,
) -> FlatEarthComparison:
    """Depth's ground range vs. what a flat earth at ``camera_height`` would predict.

    This is the question that decides how much the depth product buys a label
    estimator over plain trigonometry: if a ray depressed by theta lands at
    ``h/tan(theta)``, the payload is carrying no information the 2021 OLS fit could
    not learn from ``sv_image_y`` alone.
    """
    h, w = payload.height, payload.width
    cloud = gd.compute_point_cloud(payload).reshape(h, w, 3).astype(np.float64)
    horizontal = np.hypot(cloud[..., 0], cloud[..., 1])

    depr = depression_angles(payload)
    band = (depr > math.radians(band_deg[0])) & (depr < math.radians(band_deg[1]))
    predicted = (camera_height / np.tan(depr[band]))[:, None]
    observed = horizontal[band, :]

    idx = payload.indices.reshape(h, w)[band, :]
    valid = np.isfinite(observed) & (observed < max_range_m) & (idx != 0)
    if not valid.any():
        nan = float("nan")
        return FlatEarthComparison(0, nan, nan, nan, nan, nan, nan, nan, nan)

    residual = observed[valid] - np.broadcast_to(predicted, observed.shape)[valid]

    tilt = plane_tilt_deg(payload)[idx[valid]]
    ground_idx = ground_plane_index(payload)
    deviating = np.abs(residual) > 1.0
    n_dev = int(deviating.sum())
    if n_dev:
        dev_tilt = tilt[deviating]
        dev_idx = idx[valid][deviating]
        facade = dev_tilt >= VERTICAL_TILT_DEG
        terrain = (dev_tilt <= HORIZONTAL_TILT_DEG) & (dev_idx != ground_idx)
        f_facade = float(facade.mean())
        f_terrain = float(terrain.mean())
        f_other = float(1.0 - f_facade - f_terrain)
    else:
        f_facade = f_terrain = f_other = 0.0

    return FlatEarthComparison(
        n_pixels=int(residual.size),
        median_residual_m=float(np.median(residual)),
        p10_residual_m=float(np.percentile(residual, 10)),
        p90_residual_m=float(np.percentile(residual, 90)),
        frac_within_1m=float((np.abs(residual) < 1.0).mean()),
        frac_within_2m=float((np.abs(residual) < 2.0).mean()),
        frac_dev_facade=f_facade,
        frac_dev_terrain=f_terrain,
        frac_dev_other=f_other,
    )


# ---------- T3: what a stored label's pixel actually lands on

@dataclass
class LabelHit:
    plane_idx: int
    hit_class: str  # ground | terrain | facade | oblique | sky | out_of_bounds
    range_m: float  # euclidean ray distance, NaN if no plane
    horizontal_m: float  # horizontal distance from the camera
    height_above_ground_m: float  # camera_height - dz; ~0 at road level
    flat_earth_m: float  # h/tan(depression) for the same pixel
    flat_earth_excess_m: float  # range minus that; the "what depth adds" term
    neighbourhood_range_ratio: float  # range / median range of the 3x3 around it


def classify_label_hit(
    payload: gd.DepthPayload,
    sv_image_x: float,
    sv_image_y: float,
    camera_height: float,
) -> LabelHit:
    """Classify the depth hit for one stored label, using the v6 pixel lookup.

    The pixel is chosen exactly as ``gsv_depth.v6_to_latlng`` chooses it (the same
    ``ceil`` and the same seam-wrap behaviour), so this describes the pixel that
    actually produced the stored coordinate.
    """
    h, w = payload.height, payload.width
    cloud = gd.compute_point_cloud(payload)
    out = gd.v6_to_latlng(sv_image_x, sv_image_y, 0.0, 0.0, cloud)
    nan = float("nan")

    if out.out_of_bounds:
        return LabelHit(-1, "out_of_bounds", nan, nan, nan, nan, nan, nan)

    flat_index = out.ceil_px + w * out.ceil_py
    plane_idx = int(payload.indices[flat_index])
    if plane_idx == 0:
        return LabelHit(0, "sky", nan, nan, nan, nan, nan, nan)

    rng = math.sqrt(out.dx ** 2 + out.dy ** 2 + out.dz ** 2)
    horiz = math.hypot(out.dx, out.dy)

    tilt = float(plane_tilt_deg(payload)[plane_idx])
    ground_idx = ground_plane_index(payload)
    if plane_idx == ground_idx:
        hit_class = "ground"
    elif tilt <= HORIZONTAL_TILT_DEG:
        hit_class = "terrain"
    elif tilt >= VERTICAL_TILT_DEG:
        hit_class = "facade"
    else:
        hit_class = "oblique"

    # The flat-earth counterfactual for this exact ray.
    depr = float(depression_angles(payload)[min(out.ceil_py, h - 1)])
    if depr > math.radians(1.0) and math.isfinite(camera_height):
        flat = camera_height / math.tan(depr)
    else:
        flat = nan

    t = gd.compute_depth_t(payload)
    y0, y1 = max(out.ceil_py - 1, 0), min(out.ceil_py + 2, h)
    x0, x1 = max(out.ceil_px - 1, 0), min(out.ceil_px + 2, w)
    window = t[y0:y1, x0:x1]
    finite = window[np.isfinite(window)]
    ratio = float(rng / np.median(finite)) if finite.size else nan

    return LabelHit(
        plane_idx=plane_idx,
        hit_class=hit_class,
        range_m=rng,
        horizontal_m=horiz,
        height_above_ground_m=camera_height - out.dz,
        flat_earth_m=flat,
        flat_earth_excess_m=(horiz - flat) if math.isfinite(flat) else nan,
        neighbourhood_range_ratio=ratio,
    )


def curb_height_bias_m(distance_m: float, camera_height: float, curb_m: float = 0.15) -> float:
    """Systematic overshoot from modelling a raised feature as road surface.

    A curb ramp sits ~0.15 m above the road the terrain model represents, so a ray
    aimed at its lip passes over it and lands further out: to first order the range
    grows by ``curb * d / h``. At d = 9 m and h = 2.4 m that is ~0.56 m -- a bias, not
    noise, and about a third of the deployed estimator's 1.47 m median error.
    """
    if not (math.isfinite(distance_m) and camera_height > 0):
        return float("nan")
    return curb_m * distance_m / camera_height


# ---------- T1: registering depth against the panorama's own imagery

FRAME_CONTROLS = ("identity", "x_mirror", "rotate_180", "row_flip")


def apply_frame_control(grid: np.ndarray, control: str) -> np.ndarray:
    """Re-express a depth-derived raster under a deliberately wrong frame convention.

    These are the same wrong frames the #4 pilot scored numerically (x-mirror, 180
    degree rotation, row-flip); here they become the null hypotheses for registration.
    A correct pipeline must beat all of them.
    """
    if control == "identity":
        return grid
    if control == "x_mirror":
        return grid[:, ::-1]
    if control == "rotate_180":
        return grid[:, ::-1][::-1, :]
    if control == "row_flip":
        return grid[::-1, :]
    raise ValueError(f"unknown frame control {control!r}")


def resample_to_image(grid: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resample of a depth-grid raster onto the equirect image grid.

    Both rasters cover the whole sphere with the same origin, so the mapping is a pure
    scale in normalized coordinates -- no offset, no rotation. That is precisely the
    claim the registration scores are testing, so it is applied here and challenged
    there, never assumed to hold.
    """
    h, w = grid.shape
    ys = (np.arange(height) * (h / height)).astype(np.int64).clip(0, h - 1)
    xs = (np.arange(width) * (w / width)).astype(np.int64).clip(0, w - 1)
    return grid[ys[:, None], xs[None, :]]


SKY_ERODE_ITERATIONS = 3


def sky_mask_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Conservative sky mask from imagery alone: bright, not warm, reaching the zenith.

    Deliberately crude and fixed -- never tuned per panorama. Precision matters far
    more than recall here, because the mask is used to ask "did the model put a
    surface where the image is *certainly* sky?"; a false-positive sky pixel invents a
    violation, while a missed one costs nothing. Hence the final erosion, which pulls
    the mask back from boundaries and severs the thin leaks the flood fill makes
    through foliage.

    It does not need to be an excellent segmenter: every frame control is scored
    against the SAME mask, so its failures cost all hypotheses equally and cannot
    manufacture a win for the identity frame.
    """
    from scipy import ndimage

    rgb = rgb.astype(np.int16)
    r, b = rgb[..., 0], rgb[..., 2]
    brightness = rgb.max(axis=2)
    candidate = (b >= r - 5) & (brightness >= 100)  # blue or overcast-white, not warm

    # Sky is the component that reaches the top row; bright pavement never does.
    labelled, n = ndimage.label(candidate)
    if n == 0:
        return np.zeros(candidate.shape, dtype=bool)
    touching = np.unique(labelled[0, :])
    touching = touching[touching > 0]
    if touching.size == 0:
        return np.zeros(candidate.shape, dtype=bool)
    mask = np.isin(labelled, touching)
    return ndimage.binary_erosion(mask, iterations=SKY_ERODE_ITERATIONS)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)


def plane_boundary_map(indices_grid: np.ndarray) -> np.ndarray:
    """Pixels where the plane assignment changes -- the model's own edges."""
    boundary = np.zeros(indices_grid.shape, dtype=bool)
    boundary[:, :-1] |= indices_grid[:, :-1] != indices_grid[:, 1:]
    boundary[:-1, :] |= indices_grid[:-1, :] != indices_grid[1:, :]
    return boundary


def sky_violation(no_plane_grid: np.ndarray, sky: np.ndarray) -> float:
    """Fraction of certain-sky pixels onto which the model placed a surface.

    The primary registration statistic, and it is one-sided on purpose. The model
    omits everything that is not terrain or a building footprint, so its no-plane
    region is a *superset* of the true sky: trees, wires and poles all read as
    no-plane. Counting the reverse direction as error would therefore penalize the
    correct frame for the model's foliage blindness. The only unambiguous mistake is
    a surface placed over open sky -- so that alone is scored. Lower is better; 0 is
    perfect.
    """
    total = int(sky.sum())
    if total == 0:
        return float("nan")
    return float((sky & ~no_plane_grid).sum() / total)


def structure_fraction(payload: gd.DepthPayload) -> float:
    """Fraction of above-horizon pixels the model covers with a surface.

    The registration test's statistical power, measured before the test is run. The
    model contains only terrain and building footprints, so above the horizon it holds
    buildings and nothing else. On a suburban street with no tall structures the
    no-plane region is simply "everything above the horizon" for the true frame and
    for every wrong one alike, and no sky-based statistic can separate them. Reporting
    a panorama like that as a registration failure would be wrong; it is a panorama
    with no evidence either way, and this is how they are identified.
    """
    h, w = payload.height, payload.width
    above = payload.indices.reshape(h, w)[: h // 2, :]
    return float((above != 0).mean())


def violation_against(
    payload: gd.DepthPayload, sky: np.ndarray, control: str = "identity"
) -> float:
    """Sky violation of one payload against an already-computed sky mask.

    Split out from ``registration_scores`` so a permutation null can reuse one mask
    across hundreds of pairings without re-segmenting the imagery each time.
    """
    height, width = sky.shape
    idx = payload.indices.reshape(payload.height, payload.width)
    grid = resample_to_image(apply_frame_control(idx, control), width, height) == 0
    return sky_violation(grid, sky)


@dataclass
class RegistrationScore:
    control: str
    sky_violation: float  # primary; lower is better
    sky_iou: float  # secondary; catches the vertical flips outright


def registration_scores(
    payload: gd.DepthPayload,
    rgb: np.ndarray,
    controls: tuple[str, ...] = FRAME_CONTROLS,
) -> list[RegistrationScore]:
    """Score depth-to-imagery registration under the identity frame and its rivals.

    ``rgb`` is an equirectangular panorama image fetched from a *different Google
    host* than the depth payload. Passing this panorama's own imagery scores the frame
    conventions; passing a different panorama's imagery gives the shuffle control,
    which is the one that tests whether the payload is bound to this scene at all.
    """
    height, width = rgb.shape[:2]
    sky = sky_mask_from_rgb(rgb)
    idx = payload.indices.reshape(payload.height, payload.width)

    out = []
    for control in controls:
        grid = resample_to_image(apply_frame_control(idx, control), width, height) == 0
        out.append(RegistrationScore(
            control=control,
            sky_violation=sky_violation(grid, sky),
            sky_iou=_iou(grid, sky),
        ))
    return out


def column_offset_sweep(
    payload: gd.DepthPayload, rgb: np.ndarray, max_shift: int = 128, step: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Sky violation as a function of horizontal shift, in depth columns.

    The sharpest available test of the "no rotation offset" claim: if depth column c
    really corresponds to image column 26c, this bottoms out at exactly 0. A minimum
    anywhere else would mean the client has been reading the scene rotated -- an error
    the #4 pilot's self-comparison structurally could not have detected, since it
    compared the pipeline against coordinates the same pipeline wrote.
    """
    height, width = rgb.shape[:2]
    sky = sky_mask_from_rgb(rgb)
    idx = payload.indices.reshape(payload.height, payload.width)

    shifts = np.arange(-max_shift, max_shift + 1, step)
    scores = np.array([
        sky_violation(resample_to_image(np.roll(idx, int(s), axis=1), width, height) == 0, sky)
        for s in shifts
    ])
    return shifts, scores


# ---------- imagery: verbatim tiles in, equirect array out

def stitch_tiles(tiles: list[dict], width: int, height: int) -> np.ndarray:
    """Assemble verbatim tile JPEGs into one equirectangular RGB array.

    ``tiles`` carries the bytes exactly as Google served them (see
    ``data/depth-validation-tiles.jsonl.gz``), so this is the only step between the
    committed artifact and every image-based number in the report -- and it is
    deterministic, which is what makes the whole analysis replayable offline.
    """
    from PIL import Image

    canvas = Image.new("RGB", (width, height))
    for tile in sorted(tiles, key=lambda t: (t["y"], t["x"])):
        img = Image.open(io.BytesIO(tile["bytes"]))
        canvas.paste(img, (tile["x"] * img.width, tile["y"] * img.height))
    return np.asarray(canvas)


def label_pixel_in_image(
    sv_image_x: float,
    sv_image_y: float,
    width: int,
    height: int,
    pano_yaw_deg: float,
) -> tuple[int, int]:
    """Where a stored label sits in an equirectangular panorama image.

    Two coordinate frames are in play and they are NOT the same, which is the whole
    point of this function:

    - ``sv_image_x`` is **north-referenced**: ``sv_image_x / 13312 * 360`` is the
      label's true compass bearing. Verified over all 395,147 cleaned labels against
      the independently recorded POV ``heading`` (centred on 0, 100% within 60 deg;
      the heading-shifted alternative keeps only 32%).
    - The **panorama raster is heading-centred**: column 0 is bearing
      ``pano_yaw - 180``, so the vehicle's forward direction sits at image centre.
      This is Project Sidewalk's own documented convention, from the 2017
      ``GSVImage.py``: ``heading = 360 * (x / width) + (pano_yaw_deg - 180)``. It is
      also what ``current_pano_x`` uses, and it is what road links, building facades
      and depth sightlines all confirm.

    So placing a label on the imagery needs the yaw rotation; using ``sv_image_x``
    directly displaces it by ``(180 - pano_yaw) / 360`` of the image width -- up to
    half a panorama, and zero only for panoramas that happen to face due south.
    """
    bearing = (sv_image_x * gd.SV_IMAGE_SCALE / gd.DEPTH_W) * 360.0
    fx = ((bearing - pano_yaw_deg + 180.0) % 360.0) / 360.0
    fy = (gd.SV_IMAGE_Y_ORIGIN - sv_image_y) * gd.SV_IMAGE_SCALE / gd.DEPTH_H
    return (
        int(np.clip(fx * width, 0, width - 1)),
        int(np.clip(fy * height, 0, height - 1)),
    )
