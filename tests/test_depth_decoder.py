"""Unit tests for python/gsv_depth.py -- the v6-replica depth decoder and lookup.

Layer 1 of the depth-pilot suite: everything here is offline and synthetic. The
fixture payload is hand-built so every expected value is derivable on paper; a
committed real payload (tests/fixtures/depth-pilot/) anchors the same decoder
against actual Google bytes in test_real_payload below.

Assertion policy: exact equality only within self-consistent code paths (the
decoder tested against bytes this file wrote); float comparisons against math
done here use 1-ulp-scale tolerances, never bitwise-vs-JS expectations.
"""

import base64
import gzip
import json
import math
import os
import struct
import zlib

import numpy as np
import pytest

from gsv_depth import (
    DEFAULT_CAMERA_HEIGHT,
    DEPTH_H,
    DEPTH_W,
    NO_PLANE_VALUE,
    RecomputedLabel,
    camera_height_qc,
    compute_depth_t,
    compute_point_cloud,
    decode_depth_payload,
    extract_depth_b64,
    is_on_f32_grid,
    ulp32,
    ulp32_distance,
    v6_to_latlng,
)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "depth-pilot")


# ---------- synthetic payload construction

def build_payload(planes, index_fn, w=DEPTH_W, h=DEPTH_H, compress=False):
    """Assemble payload bytes the way GSVPanoPointCloud.js parses them.

    planes: list of (n3, d) including the index-0 null plane.
    index_fn(x, y) -> plane index per payload pixel.
    """
    offset = 9  # clean layout for synthetic data; real payloads use offset=8
    header = struct.pack("<BHHHH", 8, len(planes), w, h, offset)
    indices = bytes(index_fn(x, y) for y in range(h) for x in range(w))
    plane_bytes = b"".join(
        struct.pack("<ffff", n[0], n[1], n[2], d) for n, d in planes
    )
    raw = header + indices + plane_bytes
    if compress:
        raw = zlib.compress(raw)
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


GROUND_H = 2.2

# Plane 1: flat ground 2.2 m below camera. Plane 2: wall 10 m north (n points
# south toward the camera; d is the perpendicular distance).
PLANES = [
    ((0.0, 0.0, 0.0), 0.0),  # index 0: no plane
    ((0.0, 0.0, -1.0), GROUND_H),
    ((0.0, -1.0, 0.0), 10.0),
]

NO_PLANE_X_RANGE = range(100, 110)  # a below-horizon hole for no-plane tests


def synthetic_index(x, y):
    # z-down frame: payload rows y >= h/2 point below the horizon (ground),
    # rows y < h/2 above it (wall).
    if y >= DEPTH_H // 2:
        return 0 if x in NO_PLANE_X_RANGE else 1
    return 2


@pytest.fixture(scope="module")
def payload():
    return decode_depth_payload(build_payload(PLANES, synthetic_index))


@pytest.fixture(scope="module")
def cloud(payload):
    return compute_point_cloud(payload)


# ---------- decoding

def test_header_roundtrip(payload):
    assert payload.header_size == 8
    assert payload.n_planes == 3
    assert (payload.width, payload.height) == (DEPTH_W, DEPTH_H)
    assert not payload.was_compressed


def test_planes_roundtrip(payload):
    assert payload.planes_n.shape == (3, 3)
    assert payload.planes_n.dtype == np.float32
    assert tuple(payload.planes_n[1]) == (0.0, 0.0, -1.0)
    assert float(payload.planes_d[1]) == np.float32(GROUND_H)


def test_indices_roundtrip(payload):
    idx = payload.indices.reshape(DEPTH_H, DEPTH_W)
    assert idx[DEPTH_H - 1, 0] == 1  # below-horizon ground (bottom rows)
    assert idx[130, 105] == 0  # the hole
    assert idx[0, 0] == 2  # above-horizon wall (top rows)


def test_zlib_wrapped_payload_decodes_identically():
    plain = decode_depth_payload(build_payload(PLANES, synthetic_index))
    wrapped = decode_depth_payload(build_payload(PLANES, synthetic_index, compress=True))
    assert wrapped.was_compressed
    assert np.array_equal(plain.indices, wrapped.indices)
    assert np.array_equal(plain.planes_n, wrapped.planes_n)
    assert np.array_equal(plain.planes_d, wrapped.planes_d)


def test_unpadded_urlsafe_b64_accepted():
    s = build_payload(PLANES, synthetic_index)
    assert "=" not in s  # the endpoint serves unpadded URL-safe base64
    decode_depth_payload(s)


def test_bad_header_raises():
    with pytest.raises((ValueError, zlib.error)):
        decode_depth_payload(base64.urlsafe_b64encode(b"\x07nonsense").decode())


# ---------- point-cloud geometry

def payload_angles(x, y):
    """The v6 direction for payload pixel (x, y): (polar-from-nadir phi, azimuth theta)."""
    phi = (DEPTH_H - y - 0.5) / DEPTH_H * math.pi
    theta = (DEPTH_W - x - 0.5) / DEPTH_W * 2 * math.pi + math.pi / 2
    return phi, theta


def test_ground_pixels_sit_at_plus_camera_height(cloud):
    # z is DOWN: ground points carry z = +camera_height.
    c = cloud.reshape(DEPTH_H, DEPTH_W, 3)
    ys = [130, 170, 210, 250]  # below-horizon rows, near-horizon to nadir
    for y in ys:
        z = c[y, 200, 2]  # x=200 avoids the no-plane hole
        assert z == pytest.approx(GROUND_H, abs=1e-5)


def test_ray_distance_matches_h_over_sin_depression(payload):
    t = compute_depth_t(payload)
    for y in [135, 160, 200]:
        phi, _ = payload_angles(0, y)
        depression = math.pi / 2 - phi  # >0 below the horizon
        expected = GROUND_H / math.sin(depression)
        assert t[y, 200] == pytest.approx(expected, rel=1e-5)


def test_wall_pixel_distance(payload):
    # Wall n=(0,-1,0), d=10: t = 10/|v_y|. Pick an above-horizon pixel and check.
    t = compute_depth_t(payload)
    y, x = 50, 300
    phi, theta = payload_angles(x, y)
    vy = math.sin(phi) * math.sin(theta)
    assert t[y, x] == pytest.approx(10.0 / abs(vy), rel=1e-5)


def test_no_plane_pixels_are_1e19(cloud):
    c = cloud.reshape(DEPTH_H, DEPTH_W, 3)
    assert c[130, 105, 0] == np.float32(NO_PLANE_VALUE)
    assert c[130, 105, 1] == np.float32(NO_PLANE_VALUE)


def test_cloud_is_float32_flat(cloud):
    assert cloud.dtype == np.float32
    assert cloud.shape == (3 * DEPTH_W * DEPTH_H,)


# ---------- v6 label lookup (Label.js toLatLng)

PANO_LAT, PANO_LNG = 47.6, -122.3


def expected_lookup(sv_x, sv_y, cloud):
    """The lookup spelled out longhand, mirroring the JS statement by statement."""
    r = 1.0 / 26.0
    cx = math.ceil(sv_x * r)
    cy = math.ceil((3328 - sv_y) * r)
    flat = 3 * (cx + 512 * cy)
    dx, dy = float(cloud[flat]), float(cloud[flat + 1])
    lat = PANO_LAT + dy / 111111.0
    lng = PANO_LNG + dx / (111111.0 * math.cos(math.radians(PANO_LAT)))
    return cx, cy, dx, dy, lat, lng


def test_basic_lookup_matches_longhand(cloud):
    res = v6_to_latlng(6656, -455, PANO_LAT, PANO_LNG, cloud)
    cx, cy, dx, dy, lat, lng = expected_lookup(6656, -455, cloud)
    assert (res.ceil_px, res.ceil_py) == (cx, cy) == (256, 146)
    assert (res.dx, res.dy) == (dx, dy)
    assert (res.lat, res.lng) == (lat, lng)
    assert not (res.no_plane or res.seam_wrap or res.out_of_bounds)


def test_below_horizon_label_lands_on_ground_plane(cloud):
    # sv_image_y = -455 is 12.3 deg below the horizon -> payload row 146, which
    # must hit the ground plane: dz = +GROUND_H in the z-down frame.
    res = v6_to_latlng(6656, -455, PANO_LAT, PANO_LNG, cloud)
    assert res.dz == pytest.approx(GROUND_H, abs=1e-5)


def test_seam_wrap_reads_next_row(cloud):
    # ceil(13300/26) = 512: the flat index is identical to (x=0, row cy+1) --
    # the JS silently walked into the next raster row and so do we.
    res = v6_to_latlng(13300, -455, PANO_LAT, PANO_LNG, cloud)
    assert res.seam_wrap
    assert res.ceil_px == 512
    wrapped_flat = 3 * (0 + 512 * (res.ceil_py + 1))
    assert res.dx == float(cloud[wrapped_flat])


def test_dc_overflow_reads_far_row(cloud):
    # DC legacy rows reach sv_image_x = 14864: ceil(14864/26) = 572, which lands
    # on (x=60, row cy+1) of the raster.
    res = v6_to_latlng(14864, -455, PANO_LAT, PANO_LNG, cloud)
    assert res.seam_wrap
    assert res.ceil_px == 572
    wrapped_flat = 3 * (60 + 512 * (res.ceil_py + 1))
    assert res.dx == float(cloud[wrapped_flat])


def test_out_of_bounds_returns_nan():
    # sv_image_y = -3328 -> ceil_py = 256, one row past the raster: JS reads
    # undefined and produces NaN, never an exception.
    cloud = np.zeros(3 * DEPTH_W * DEPTH_H, dtype=np.float32)
    res = v6_to_latlng(6656, -3328, PANO_LAT, PANO_LNG, cloud)
    assert res.out_of_bounds
    assert math.isnan(res.lat) and math.isnan(res.lng)


def test_no_plane_label_is_flagged_and_absurd(cloud):
    # x = 2730 -> ceil(2730/26) = 105, inside the hole; the 1e19 sentinel makes
    # the offset astronomically large, exactly like the absurd stored rows.
    res = v6_to_latlng(2730, -455, PANO_LAT, PANO_LNG, cloud)
    assert res.no_plane
    assert abs(res.lat) > 1e10


def test_scale_uses_multiplication_by_reciprocal():
    # scaleImageCoordinate multiplies by r = 1/26 (a rounded double); dividing by
    # 26 is NOT the same operation at the ulp level, and ceil() sits on those
    # boundaries. Guard the exact expression.
    r = 1.0 / 26.0
    for sv_x in [0, 1, 25, 26, 6656, 13286, 13311, 13312, 14864]:
        assert math.ceil(sv_x * r) == v6_to_latlng(
            sv_x, 0, PANO_LAT, PANO_LNG, np.zeros(3 * DEPTH_W * DEPTH_H, np.float32)
        ).ceil_px


# ---------- camera-height QC

def test_qc_finds_ground_plane(payload):
    qc = camera_height_qc(payload)
    assert qc.n_planes == 3
    assert qc.ground_plane_idx == 1
    assert qc.ground_d == pytest.approx(GROUND_H, abs=1e-6)
    assert qc.ground_height == pytest.approx(GROUND_H, abs=1e-6)
    assert qc.ground_tilt_deg == pytest.approx(0.0, abs=1e-6)
    assert qc.ground_pixel_share > 0.9
    assert not qc.is_default


def test_qc_band_median_agrees_with_plane(payload):
    qc = camera_height_qc(payload)
    # On synthetic flat ground the ray cross-check recovers the plane height
    # to float32 precision, and the residual is ~0.
    assert qc.band_height_median == pytest.approx(GROUND_H, abs=1e-4)
    assert qc.band_height_mad < 1e-4


def test_qc_detects_google_default_structurally():
    default_planes = [((0.0, 0.0, 0.0), 0.0), ((0.0, 0.0, -1.0), DEFAULT_CAMERA_HEIGHT)]
    p = decode_depth_payload(
        build_payload(default_planes, lambda x, y: 1 if y >= DEPTH_H // 2 else 0)
    )
    assert camera_height_qc(p).is_default

    # Same two-plane shape but a measured height: NOT the default.
    measured = [((0.0, 0.0, 0.0), 0.0), ((0.0, 0.0, -1.0), 2.4986)]
    p2 = decode_depth_payload(
        build_payload(measured, lambda x, y: 1 if y >= DEPTH_H // 2 else 0)
    )
    assert not camera_height_qc(p2).is_default


def test_qc_tilted_ground():
    # A 5-degree-tilted ground plane: height = d / |n_z|, tilt reported.
    tilt = math.radians(5)
    n = (0.0, math.sin(tilt), -math.cos(tilt))
    p = decode_depth_payload(
        build_payload(
            [((0.0, 0.0, 0.0), 0.0), (n, 2.0)],
            lambda x, y: 1 if y >= DEPTH_H // 2 else 0,
        )
    )
    qc = camera_height_qc(p)
    assert qc.ground_tilt_deg == pytest.approx(5.0, abs=0.01)
    assert qc.ground_height == pytest.approx(2.0 / math.cos(tilt), rel=1e-4)


# ---------- float32 storage-lattice helpers

def test_f32_grid_membership():
    assert is_on_f32_grid(38.94049072265625)  # a real stored DC latitude
    assert not is_on_f32_grid(38.94049172265625)


def test_ulp32_magnitude_at_dc():
    # One float32 ulp at DC latitude is ~3.8e-6 deg, i.e. ~0.42 m of latitude.
    u = ulp32(38.94049072265625)
    assert u * 111111 == pytest.approx(0.42, abs=0.03)


def test_ulp32_distance():
    stored = 38.94049072265625
    assert ulp32_distance(stored, stored) == 0.0
    assert ulp32_distance(stored + ulp32(stored), stored) == pytest.approx(1.0, rel=1e-6)


def test_ulp32_positive_for_negative_values():
    # np.spacing is signed; a negative longitude must still give a positive ulp
    # (this bug made every ulp distance on real data infinite).
    assert ulp32(-122.3) > 0
    assert ulp32_distance(-122.3, -122.3) == 0.0
    assert math.isfinite(ulp32_distance(-122.3000001, -122.3))


# ---------- raw photometa plumbing

def test_extract_depth_b64_path():
    # response[1][0][5][0][5][1][2] == "payload-b64"; build the nesting
    # positionally so the indices are visibly right.
    msg5 = [None] * 6
    msg5[5] = [None, [None, None, "payload-b64"]]
    msg = [None] * 6
    msg[5] = [msg5]
    response = [None, [msg]]
    assert extract_depth_b64(response) == "payload-b64"
    assert extract_depth_b64([None, [[None] * 6]]) is None
    assert extract_depth_b64(None) is None


# ---------- committed real payload (added by the calibration step)

REAL_PAYLOAD_FILE = os.path.join(FIXTURES, "real-payload.json.gz")


@pytest.mark.skipif(
    not os.path.exists(REAL_PAYLOAD_FILE), reason="real-payload fixture not committed yet"
)
def test_real_payload():
    """Anchor the decoder against actual Google bytes with hand-verified values."""
    with gzip.open(REAL_PAYLOAD_FILE, "rt", encoding="utf-8") as f:
        fx = json.load(f)
    p = decode_depth_payload(fx["depth_b64"])
    assert p.header_size == 8
    assert (p.width, p.height) == (DEPTH_W, DEPTH_H)
    assert p.n_planes == fx["expected"]["n_planes"]
    qc = camera_height_qc(p)
    assert qc.ground_d == pytest.approx(fx["expected"]["ground_d"], abs=1e-6)
    assert qc.is_default == fx["expected"]["is_default"]
    t = compute_depth_t(p)
    for probe in fx["expected"]["t_probes"]:  # [y, x, t] spot values
        assert t[probe[0], probe[1]] == pytest.approx(probe[2], rel=1e-5)
