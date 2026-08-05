"""Fresh GSV depth, decoded with the EXACT 2020 Project Sidewalk client algorithm.

This module supports the issue #4 pilot: fetch today's depth payload for a panorama
through sk-zk/streetlevel's unofficial photometa endpoint, then recompute label
positions from stored ``sv_image_x``/``sv_image_y`` with a bit-level replication of
the client code that produced the recovered ground truth (SidewalkWebpage v6.0.0,
verified against the primary source):

- ``GSVPanoDepth.js`` / ``GSVPanoPointCloud.js`` -- payload header/planes parsing and
  ``computePointCloud`` (polar ``phi = (h-y-0.5)/h*pi``, azimuth
  ``theta = (w-x-0.5)/w*2*pi + pi/2``, ``t = |d / (v.n)|``, points stored to a
  Float32Array at flat index ``3*(y*512 + x)``, NOT mirrored; no-plane pixels 1e19).
  The frame that makes the whole pipeline cohere is x=east, y=north, **z=down**:
  payload row 0 is the zenith, rows >= h/2 are below the horizon (phi is the polar
  angle from nadir), ground points carry dz = +camera_height, and Google's default
  ground plane normal (0, 0, -1) is the up-facing one. This is forced by the
  lookup: sv_image_y = -455 (12.3 deg below horizon) lands on row 146, and the
  2021 result proves those lookups hit ground, not sky.
- ``Label.js::toLatLng`` + ``Utilities.js::scaleImageCoordinate`` +
  ``UtilitiesMath.js::latlngOffset`` -- ``px = sv_image_x*(1/26)``,
  ``py = (3328 - sv_image_y)*(1/26)``, ``idx = 3*(ceil(px) + 512*ceil(py))``,
  ``dlat = dy/111111``, ``dlng = dx/(111111*cos(lat))``.

Fidelity rules (deviations would silently change the comparison):

1. Trig tables, plane parameters, and stored points are float32 exactly where v6
   used Float32Array/getFloat32, but arithmetic BETWEEN those values runs in
   float64, because JS reads a float32 array element back as a float64. NumPy
   would otherwise keep float32*float32 in float32.
2. The scale factor is ``1/26`` computed once and multiplied (not ``x/26``): the
   two differ in the last ulp, and ``ceil`` sits right on those boundaries.
3. The flat-index seam wrap is preserved: ``ceil(px) >= 512`` silently reads the
   NEXT raster row (1,642 recovered rows do this), and DC's legacy
   ``sv_image_x > 13312`` walks even further. Reads past the end of the array are
   JS ``undefined`` -> NaN, and are flagged, not raised.
4. No-plane pixels (plane index 0) yield 1e19 per component and propagate to
   absurd lat/lng exactly as production did circa 2017-2020.

The modern photometa payload is served uncompressed; the 2020 cbk payload was
zlib-deflated (v6 ran ``zpipe.inflate`` unconditionally). ``decode_depth_payload``
accepts both. Network access is confined to ``fetch_photometa_raw`` (imports
streetlevel lazily) so everything else is testable offline.
"""

from __future__ import annotations

import base64
import math
import zlib
from dataclasses import dataclass

import numpy as np

# ---------- constants (v6 names in comments)

DEPTH_W = 512  # header width on every payload observed 2011-2026
DEPTH_H = 256
NO_PLANE_VALUE = 9999999999999999999.0  # v6's literal for planeIdx == 0 pixels
SV_IMAGE_SCALE = 1.0 / 26.0  # scaleImageCoordinate r: 13312/26 = 512, 6656/26 = 256
SV_IMAGE_Y_ORIGIN = 3328  # "0 for image y-axis is at *3328*!" (Utilities.js)
METERS_PER_DEGREE = 111111.0  # latlngOffset's flat-earth constant
DEFAULT_CAMERA_HEIGHT = 2.5  # Google's fallback ground plane: exactly 2 planes,
#                              normal exactly (0, 0, -1), d exactly 2.5


# ---------- payload decoding (GSVPanoPointCloud.js parseHeader/parsePlanes)

@dataclass
class DepthPayload:
    header_size: int
    n_planes: int
    width: int
    height: int
    offset: int
    indices: np.ndarray  # uint8, flat, length width*height, row-major payload order
    planes_n: np.ndarray  # float32, shape (n_planes, 3)
    planes_d: np.ndarray  # float32, shape (n_planes,)
    was_compressed: bool  # zlib-deflated (2020 cbk style) vs raw (photometa style)


def decode_depth_payload(b64_string: str) -> DepthPayload:
    """Decode a raw depth payload (URL-safe unpadded base64, optionally deflated)."""
    b64_string = b64_string.replace("-", "+").replace("_", "/")  # v6 decode()
    b64_string += "=" * ((4 - len(b64_string) % 4) % 4)
    raw = base64.b64decode(b64_string)

    # The header always starts with headerSize == 8. photometa serves the struct
    # bare; cbk wrapped it in zlib (v6 inflated unconditionally).
    was_compressed = False
    if len(raw) == 0 or raw[0] != 8:
        raw = zlib.decompress(raw)
        was_compressed = True
    if raw[0] != 8:
        raise ValueError(f"depth payload header_size {raw[0]} != 8")

    header_size = raw[0]
    n_planes = int.from_bytes(raw[1:3], "little")
    width = int.from_bytes(raw[3:5], "little")
    height = int.from_bytes(raw[5:7], "little")
    offset = int.from_bytes(raw[7:9], "little")

    indices = np.frombuffer(raw, dtype=np.uint8, count=width * height, offset=offset)
    planes_raw = np.frombuffer(
        raw, dtype="<f4", count=n_planes * 4, offset=offset + width * height
    ).reshape(n_planes, 4)

    return DepthPayload(
        header_size=header_size,
        n_planes=n_planes,
        width=width,
        height=height,
        offset=offset,
        indices=indices,
        planes_n=np.ascontiguousarray(planes_raw[:, :3]),
        planes_d=np.ascontiguousarray(planes_raw[:, 3]),
        was_compressed=was_compressed,
    )


# ---------- point cloud (GSVPanoPointCloud.js computePointCloud, verbatim)

def compute_point_cloud(payload: DepthPayload) -> np.ndarray:
    """Flat float32 array of length 3*w*h: (x_east, y_north, z) per payload pixel.

    v6 stored point = t*v at flat index 3*(y*w + x) -- payload order, no mirror
    (unlike its computeDepthMap variant and unlike streetlevel's DepthMap, which
    both mirror x on output).
    """
    w, h = payload.width, payload.height

    # v6 kept the trig tables in Float32Arrays; reading an element back gives the
    # float32 value as a float64. Cast to f32 then up to f64 before arithmetic.
    y_idx = np.arange(h, dtype=np.float64)
    phi = (h - y_idx - 0.5) / h * np.pi  # polar from nadir: row 0 zenith, row h-1 nadir
    sin_phi = np.float32(np.sin(phi)).astype(np.float64)
    cos_phi = np.float32(np.cos(phi)).astype(np.float64)

    x_idx = np.arange(w, dtype=np.float64)
    theta = (w - x_idx - 0.5) / w * 2.0 * np.pi + np.pi / 2.0  # azimuth
    sin_theta = np.float32(np.sin(theta)).astype(np.float64)
    cos_theta = np.float32(np.cos(theta)).astype(np.float64)

    # v = (sin_phi*cos_theta, sin_phi*sin_theta, cos_phi), per pixel, float64.
    vx = sin_phi[:, None] * cos_theta[None, :]
    vy = sin_phi[:, None] * sin_theta[None, :]
    vz = np.broadcast_to(cos_phi[:, None], (h, w))

    idx = payload.indices.reshape(h, w)
    n = payload.planes_n.astype(np.float64)[idx]  # (h, w, 3), f32 values as f64
    d = payload.planes_d.astype(np.float64)[idx]  # (h, w)

    denom = vx * n[..., 0] + vy * n[..., 1] + vz * n[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.abs(d / denom)

    cloud = np.empty((h, w, 3), dtype=np.float32)
    cloud[..., 0] = t * vx  # f64 product, truncated to f32 on store, as in JS
    cloud[..., 1] = t * vy
    cloud[..., 2] = t * vz
    cloud[idx == 0] = np.float32(NO_PLANE_VALUE)
    return cloud.reshape(-1)


def compute_depth_t(payload: DepthPayload) -> np.ndarray:
    """Euclidean ray distance t per payload pixel, shape (h, w). NaN where no plane.

    Same t as v6/streetlevel (|v| == 1 so t == |point|); payload x order, unmirrored.
    """
    w, h = payload.width, payload.height
    cloud = compute_point_cloud(payload).reshape(h, w, 3).astype(np.float64)
    t = np.sqrt((cloud ** 2).sum(axis=2))
    t[payload.indices.reshape(h, w) == 0] = np.nan
    return t


# ---------- label recomputation (Label.js toLatLng, single-point path)

@dataclass
class RecomputedLabel:
    lat: float
    lng: float
    dx: float  # east meters (float32 value), 1e19 if no plane, NaN if out of bounds
    dy: float  # north meters
    dz: float
    ceil_px: int
    ceil_py: int
    no_plane: bool
    seam_wrap: bool  # ceil(px) >= 512: flat index silently reads the next row
    out_of_bounds: bool  # flat index past the array: JS undefined -> NaN


def v6_to_latlng(
    sv_image_x: float,
    sv_image_y: float,
    pano_lat: float,
    pano_lng: float,
    point_cloud: np.ndarray,
) -> RecomputedLabel:
    """Replicate Label.js toLatLng for a single-point label, quirks included."""
    # scaleImageCoordinate(x, y, 1/26): multiply by the rounded 1/26, don't divide.
    px = sv_image_x * SV_IMAGE_SCALE
    py = (SV_IMAGE_Y_ORIGIN - sv_image_y) * SV_IMAGE_SCALE
    cx = math.ceil(px)
    cy = math.ceil(py)
    flat = 3 * (cx + DEPTH_W * cy)

    if flat < 0 or flat + 2 >= point_cloud.shape[0]:
        dx = dy = dz = float("nan")
        out_of_bounds = True
    else:
        dx = float(point_cloud[flat])
        dy = float(point_cloud[flat + 1])
        dz = float(point_cloud[flat + 2])
        out_of_bounds = False

    # latlngOffset(panoLat, dx, dy)
    dlat = dy / METERS_PER_DEGREE
    dlng = dx / (METERS_PER_DEGREE * math.cos(math.radians(pano_lat)))

    return RecomputedLabel(
        lat=pano_lat + dlat,
        lng=pano_lng + dlng,
        dx=dx,
        dy=dy,
        dz=dz,
        ceil_px=cx,
        ceil_py=cy,
        no_plane=(not out_of_bounds) and abs(dx) >= 1e18,
        seam_wrap=cx >= DEPTH_W,
        out_of_bounds=out_of_bounds,
    )


# ---------- per-panorama camera-height QC (plane-derived, per issue #4 comment)

@dataclass
class CameraHeightQC:
    n_planes: int
    ground_plane_idx: int  # -1 if no near-vertical plane found
    ground_d: float  # plane offset = camera-to-plane distance along the normal
    ground_height: float  # vertical camera height = d / |n_z|
    ground_tilt_deg: float  # angle of the plane normal off vertical
    ground_pixel_share: float  # fraction of below-horizon pixels on that plane
    is_default: bool  # Google's fallback: 2 planes, n exactly (0,0,-1), d == 2.5
    band_height_median: float  # cross-check: median t*sin(depression), 20-60 deg
    band_height_mad: float  # spread of the same (plane-fit residual proxy)


def camera_height_qc(payload: DepthPayload) -> CameraHeightQC:
    """Camera height from the payload's plane list, with a ray-based cross-check.

    The ground plane's d IS the camera height when its normal is vertical: in the
    z-down frame, ground rays (v_z > 0) against n=(0,0,-1) give points at
    z = +d, i.e. d meters below the camera. The 2.500 m default that 14% of panos
    return is detected structurally -- exactly two planes (null + ground) with
    normal exactly (0,0,-1) and d exactly 2.5 -- not by value comparison.
    """
    w, h = payload.width, payload.height
    idx = payload.indices.reshape(h, w)

    # Ground candidates: near-vertical normals, weighted by below-horizon pixels.
    # Payload rows y >= h/2 point below the horizon (z-down frame, phi < pi/2).
    below = idx[h // 2 :, :]
    counts = np.bincount(below.reshape(-1), minlength=payload.n_planes).astype(float)
    counts[0] = 0.0  # index 0 is the no-plane marker
    nz = payload.planes_n[:, 2].astype(np.float64)
    vertical = np.abs(nz) > 0.95
    ground_idx = -1
    if vertical.any() and counts[vertical].sum() > 0:
        cand = np.where(vertical, counts, 0.0)
        ground_idx = int(cand.argmax())

    if ground_idx >= 0:
        d = float(payload.planes_d[ground_idx])
        g_nz = abs(float(nz[ground_idx]))
        height = d / g_nz if g_nz > 0 else float("nan")
        tilt = math.degrees(math.acos(min(g_nz, 1.0)))
        share = counts[ground_idx] / max(below.size, 1)
    else:
        d = height = tilt = float("nan")
        share = 0.0

    is_default = (
        payload.n_planes == 2
        and ground_idx == 1
        and tuple(payload.planes_n[1]) == (0.0, 0.0, -1.0)
        and float(payload.planes_d[1]) == DEFAULT_CAMERA_HEIGHT
    )

    # Cross-check on rays: depression 20-60 deg is road-dominated, clear of the
    # nadir seam and of the far field where walls take over.
    t = compute_depth_t(payload)
    y_idx = np.arange(h, dtype=np.float64)
    phi = (h - y_idx - 0.5) / h * np.pi  # polar from nadir
    depression = np.pi / 2.0 - phi  # >0 below the horizon (rows y >= h/2)
    band = (depression >= math.radians(20)) & (depression <= math.radians(60))
    implied = t[band, :] * np.sin(depression[band])[:, None]
    implied = implied[np.isfinite(implied)]
    if implied.size:
        med = float(np.median(implied))
        mad = float(np.median(np.abs(implied - med)))
    else:
        med = mad = float("nan")

    return CameraHeightQC(
        n_planes=payload.n_planes,
        ground_plane_idx=ground_idx,
        ground_d=d,
        ground_height=height,
        ground_tilt_deg=tilt,
        ground_pixel_share=float(share),
        is_default=bool(is_default),
        band_height_median=med,
        band_height_mad=mad,
    )


# ---------- float32 storage-lattice helpers (the comparison floor)

def is_on_f32_grid(value: float) -> bool:
    """True if a stored coordinate is exactly representable in float32.

    ~84% of the recovered lat/lng are: the 2020 write path truncated the final
    coordinates to float32 somewhere, so a bit-perfect recompute can differ from
    the stored value by whole float32 ulps (~0.42 m lat at DC latitudes).
    """
    return math.isfinite(value) and float(np.float32(value)) == value


def ulp32(value: float) -> float:
    """Size of one float32 ulp at this value (in the value's own units).

    np.spacing is signed (negative for negative inputs -- every longitude in the
    recovered data); the ulp is a magnitude.
    """
    return abs(float(np.spacing(np.float32(value))))


def ulp32_distance(recomputed: float, stored: float) -> float:
    """|recomputed - stored| in units of the float32 ulp at the stored value."""
    u = ulp32(stored)
    return abs(recomputed - stored) / u if u > 0 else float("inf")


# ---------- fetching (network confined here; streetlevel imported lazily)

def fetch_photometa_raw(pano_id: str, session=None) -> dict:
    """Raw photometa JSON for a pano id, depth payload included.

    The high-level streetlevel Panorama discards the raw base64 and its DepthMap
    drops the plane indices, so the pilot keeps the raw response and decodes the
    payload itself with the v6 algorithm.
    """
    from streetlevel.streetview import api

    return api.find_panorama_by_id(pano_id, download_depth=True, session=session)


def extract_depth_b64(response) -> str | None:
    """The raw depth payload string inside a photometa response, or None."""
    try:
        value = response[1][0][5][0][5][1][2]
    except (IndexError, KeyError, TypeError):
        return None
    return value if isinstance(value, str) and value else None


def extract_pano_meta(response) -> dict | None:
    """Fresh pano metadata as a flat JSON-safe dict (angles in degrees).

    Uses streetlevel's tested parser; its heading/roll come back in radians and
    pitch as radians(90 - raw), so everything is converted to plain degrees here.
    """
    from streetlevel.streetview.parse import parse_panorama_id_response

    pano = parse_panorama_id_response(response)
    if pano is None:
        return None

    date_year = date_month = None
    if pano.date is not None:
        date_year = getattr(pano.date, "year", None)
        date_month = getattr(pano.date, "month", None)

    return {
        "pano_id": pano.id,
        "lat": pano.lat,
        "lng": pano.lon,
        "heading_deg": math.degrees(pano.heading) if pano.heading is not None else None,
        "pitch_deg": math.degrees(pano.pitch) if pano.pitch is not None else None,
        "roll_deg": math.degrees(pano.roll) if pano.roll is not None else None,
        "elevation_m": pano.elevation,
        "capture_year": date_year,
        "capture_month": date_month,
        "image_sizes": ";".join(f"{s.x}x{s.y}" for s in pano.image_sizes)
        if pano.image_sizes
        else None,
        "source": pano.source,
    }
