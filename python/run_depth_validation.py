"""Issue #9: is GSV depth authentic, and is it good? Registration, characterization, failure modes.

Four subcommands, run from the repo root:

    python python/run_depth_validation.py fetch     # network: imagery tiles + partner payloads
    python python/run_depth_validation.py build     # offline: committed bytes -> artifacts
    python python/run_depth_validation.py figures   # offline: artifacts -> figures/fig9-12
    python python/run_depth_validation.py gallery   # offline: artifacts -> the overlay gallery

The #4 pilot compared fresh depth against stored label positions and got ~1 m -- but both
sides came from the same depth product, so it could only speak to transport and stability.
Worse, a systematic frame error in the 2020 client (an x-mirror, say) would have passed that
test perfectly, because the pipeline was being compared against coordinates it had written
itself. This run breaks the circle by bringing in evidence the depth pipeline never touched:

  T1  registration   depth (maps.googleapis.com photometa) vs the panorama's own imagery
                     (streetviewpixels-pa.googleapis.com) -- two independent Google hosts,
                     scored against deliberately wrong frames and against other panoramas.
  T2  what it IS     the plane set's own statistics: tilt, inventory, sky structure, and how
                     much the depth departs from naive h/tan(depression).
  T3  failure modes  what each stored label's pixel actually lands on, and the error that
                     implies -- occlusion by unmodelled objects, curb-height overshoot.
  T4  cross-vintage  the same location captured years apart carries an independently rebuilt
                     model; disagreement bounds camera pose plus model error.

Replication is the point: `fetch` writes verbatim tile bytes into the committed
data/depth-validation-tiles.jsonl.gz, and `build` runs from that file plus the #4 payloads
with no network at all. Delete the cache and every number and figure still regenerates.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import depth_validation as dv  # noqa: E402
import gsv_depth as gd  # noqa: E402
from label_latlng_estimation import haversine_m  # noqa: E402
from run_depth_pilot import Throttle, write_csv_gz, write_gz_bytes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 666  # the repo's sampling seed everywhere

# Sample sizes. Deliberately modest: the registration test is a paired comparison
# (identity vs its own controls, on the same panorama and the same sky mask), so 60
# panoramas is a 60-fold replication, not a sample of 60 observations.
SCORING_N_A = 40  # 2017-2020 panos from the #4 Part A sample; these carry the labels
SCORING_N_B = 20  # modern panos from Part B; check nothing changed at 16384x8192
ADJUDICATION_N = 24  # subset re-fetched larger, for the gallery and occlusion verdicts
ADJUDICATION_FROM_B = 6  # of those, modern Part B panos, so the gallery shows both eras
CROSS_VINTAGE_N = 30  # scoring panos to pair with a historical capture of the same spot

# Zoom INDEX into pano.image_sizes, not a pixel size: the same index means 1024x512 on a
# modern 16384-wide pano but 832x416 on a 2017-era 13312-wide one. Everything downstream
# works in normalized coordinates, so the difference does not matter -- but it does mean
# the committed artifact must record each panorama's actual pixel dimensions.
SCORING_ZOOM = 1
ADJUDICATION_ZOOM = 2

CROSS_VINTAGE_MIN_YEAR_GAP = 3  # a partner from the same year is the same model rebuild
CROSS_VINTAGE_SAMPLE_STRIDE = 7  # subsample of depth pixels used in the projection test
CROSS_VINTAGE_MAX_RANGE_M = 25.0  # beyond this the baseline parallax dominates

# The registration null: each payload is scored against its own panorama's imagery and
# against PERMUTATION_K other panoramas' imagery. A single mismatched partner is a coin
# flip; a null distribution gives the true pairing a rank, which is a real test.
PERMUTATION_K = 10
# Minimum above-horizon modelled area for a panorama to carry registration evidence at
# all. Below this the model is bare ground and a horizon, which every frame reproduces.
STRUCTURE_MIN = 0.02

TILE_URL = (
    "https://streetviewpixels-pa.googleapis.com/v1/tile"
    "?cb_client=maps_sv.tactile&panoid={pano_id}&x={x}&y={y}&zoom={zoom}"
)
TILE_HEADERS = {
    "Origin": "https://www.google.com",
    "Referer": "https://www.google.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 11.0; Win64; x64; rv:151.0) Gecko/20100101 Firefox/151.0"
    ),
}


# ---------------------------------------------------------------------------- cache

def cache_dirs(cache_root):
    tiles = os.path.join(cache_root, "tiles")
    photometa = os.path.join(cache_root, "photometa")
    os.makedirs(tiles, exist_ok=True)
    os.makedirs(photometa, exist_ok=True)
    return tiles, photometa


def pilot_photometa_dir(data_dir):
    """The #4 pilot's cache, read-only: 806 responses we should never refetch."""
    return os.path.join(data_dir, "depth-pilot-cache", "photometa")


def load_photometa(pano_id, cache_root, data_dir):
    """A cached photometa response, preferring the pilot's cache. None if absent."""
    _, photometa = cache_dirs(cache_root)
    for path in (
        os.path.join(pilot_photometa_dir(data_dir), pano_id + ".json"),
        os.path.join(photometa, pano_id + ".json"),
    ):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def fetch_photometa_cached(pano_id, cache_root, data_dir, throttle, session):
    cached = load_photometa(pano_id, cache_root, data_dir)
    if cached is not None:
        return cached, False
    _, photometa = cache_dirs(cache_root)
    last_err = None
    for backoff in (0, 2, 8):
        if backoff:
            time.sleep(backoff)
        throttle.wait()
        try:
            resp = gd.fetch_photometa_raw(pano_id, session=session)
            break
        except Exception as e:  # transient network / JSON hiccups
            last_err = e
    else:
        raise RuntimeError(f"photometa fetch failed for {pano_id}: {last_err}")
    with open(os.path.join(photometa, pano_id + ".json"), "w", encoding="utf-8") as f:
        json.dump(resp, f)
    return resp, True


def fetch_tile_cached(pano_id, zoom, x, y, cache_root, throttle, session):
    """One verbatim tile. Bytes are stored and committed exactly as served."""
    tiles, _ = cache_dirs(cache_root)
    path = os.path.join(tiles, f"{pano_id}_{zoom}_{x}_{y}.jpg")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read(), False
    url = TILE_URL.format(pano_id=pano_id, x=x, y=y, zoom=zoom)
    last_err = None
    for backoff in (0, 2, 8):
        if backoff:
            time.sleep(backoff)
        throttle.wait()
        try:
            resp = session.get(url, headers=TILE_HEADERS, timeout=30)
            resp.raise_for_status()
            content = resp.content
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"tile fetch failed for {pano_id} z{zoom} {x},{y}: {last_err}")
    with open(path, "wb") as f:
        f.write(content)
    return content, True


# ---------------------------------------------------------------------------- metadata

def parse_meta(resp):
    """streetlevel's parsed panorama, or None if the id no longer resolves."""
    from streetlevel.streetview.parse import parse_panorama_id_response

    try:
        return parse_panorama_id_response(resp)
    except Exception:
        return None


def tile_grid(pano, zoom):
    """(width, height, cols, rows, tile_w, tile_h) for a zoom INDEX."""
    zoom = max(0, min(zoom, len(pano.image_sizes) - 1))
    size = pano.image_sizes[zoom]
    tw, th = pano.tile_size.x, pano.tile_size.y
    return size.x, size.y, math.ceil(size.x / tw), math.ceil(size.y / th), tw, th


# ---------------------------------------------------------------------------- sampling

def committed_sample_path(data_dir):
    return os.path.join(data_dir, "depth-validation-sample.json")


def load_committed_sample(data_dir):
    """The pano ids actually fetched, or None. This is the archival record of the draw."""
    path = committed_sample_path(data_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def choose_samples(data_dir):
    """Deterministic scoring / adjudication / cross-vintage sets from the #4 artifacts.

    Part A panos are stratified by payload class so the registration test is not run
    only on the bit-stable ones -- a panorama whose payload has drifted since 2020
    should still register against today's imagery, and that is worth checking.
    """
    panos = pd.read_csv(os.path.join(data_dir, "depth-pilot-panos.csv.gz"))
    labels = pd.read_csv(os.path.join(data_dir, "depth-pilot-labels.csv.gz"))
    rng = np.random.default_rng(SEED)

    a = panos[(panos["part"] == "a") & (panos["status"] == "ok")].copy()
    a = a.sort_values("pano_id").reset_index(drop=True)
    picks = []
    for cls, share in (("unchanged", 0.35), ("mostly_unchanged", 0.20), ("changed", 0.45)):
        pool = a[a["pano_class"] == cls]["pano_id"].to_numpy()
        n = min(len(pool), int(round(SCORING_N_A * share)))
        picks.extend(rng.choice(pool, size=n, replace=False).tolist())
    scoring_a = sorted(picks)

    b = panos[(panos["part"] == "b") & (panos["status"] == "ok")]
    b = b.sort_values("pano_id")["pano_id"].dropna().unique()
    scoring_b = sorted(rng.choice(b, size=min(SCORING_N_B, len(b)), replace=False).tolist())

    # Adjudication favours panos carrying the most labels: the occlusion question is
    # about labels, so the larger imagery should go where the labels are.
    counts = labels[labels["pano_id"].isin(scoring_a)].groupby("pano_id").size()
    ranked = counts.sort_values(ascending=False, kind="mergesort")
    adjudication = sorted(ranked.index[: ADJUDICATION_N - ADJUDICATION_FROM_B].tolist())
    adjudication += sorted(scoring_b[:ADJUDICATION_FROM_B])

    cross = sorted(scoring_a)[:CROSS_VINTAGE_N]
    return {
        "scoring_a": scoring_a,
        "scoring_b": scoring_b,
        "adjudication": sorted(set(adjudication)),
        "cross_vintage": cross,
    }


# ---------------------------------------------------------------------------- fetch

def cmd_fetch(args):
    import requests

    sample = choose_samples(args.data_dir)
    os.makedirs(args.cache_dir, exist_ok=True)
    for path in (os.path.join(args.cache_dir, "sample.json"), committed_sample_path(args.data_dir)):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(sample, f, indent=2, sort_keys=True)
            f.write("\n")

    throttle = Throttle(args.rps)
    session = requests.Session()
    scoring = sample["scoring_a"] + sample["scoring_b"]

    n_tiles = n_new = 0
    for pano_ids, zoom, name in (
        (scoring, SCORING_ZOOM, "scoring"),
        (sample["adjudication"], ADJUDICATION_ZOOM, "adjudication"),
    ):
        for pid in sorted(pano_ids):
            # Prefer the pilot's cached photometa (fetch_photometa_cached reads it
            # first), but fall back to the network so a fresh clone can refetch too.
            try:
                resp, _ = fetch_photometa_cached(
                    pid, args.cache_dir, args.data_dir, throttle, session
                )
            except RuntimeError as e:
                print(f"  [{name}] {pid}: {e}, skipped")
                continue
            pano = parse_meta(resp)
            if pano is None or not pano.image_sizes:
                print(f"  [{name}] {pid}: no usable metadata, skipped")
                continue
            _, _, cols, rows, _, _ = tile_grid(pano, zoom)
            for x in range(cols):
                for y in range(rows):
                    _, fetched = fetch_tile_cached(
                        pid, zoom, x, y, args.cache_dir, throttle, session
                    )
                    n_tiles += 1
                    n_new += int(fetched)
            print(f"  [{name}] {pid} z{zoom}: {cols}x{rows} tiles")

    # Cross-vintage partners: a historical capture of the same spot, as far back as
    # the year gap allows. Only its photometa is needed -- depth, not imagery.
    partners = {}
    for pid in sample["cross_vintage"]:
        try:
            resp, _ = fetch_photometa_cached(
                pid, args.cache_dir, args.data_dir, throttle, session
            )
        except RuntimeError:
            continue
        pano = parse_meta(resp)
        if pano is None or not pano.historical:
            continue
        year = pano.date.year if pano.date else None
        best = None
        for hist in pano.historical:
            hyear = hist.date.year if hist.date else None
            if year is None or hyear is None:
                continue
            if abs(hyear - year) >= CROSS_VINTAGE_MIN_YEAR_GAP:
                if best is None or abs(hyear - year) > best[1]:
                    best = (hist.id, abs(hyear - year))
        if best is None:
            continue
        partners[pid] = best[0]
        fetch_photometa_cached(best[0], args.cache_dir, args.data_dir, throttle, session)
        print(f"  [cross-vintage] {pid} -> {best[0]} ({best[1]} yr gap)")

    with open(os.path.join(args.cache_dir, "partners.json"), "w", encoding="utf-8") as f:
        json.dump(partners, f, indent=2, sort_keys=True)

    print(f"\n{n_tiles} tiles referenced, {n_new} newly fetched; {len(partners)} partners")
    print(f"Cache: {args.cache_dir}. Now run `build` to write the committed artifacts.")


# ---------------------------------------------------------------------------- artifacts

def _guard_artifact_shrink(path, new_count, count_existing):
    """Refuse to overwrite a committed artifact with fewer records.

    A shrink here almost always means a partial fetch cache (an interrupted `fetch`),
    and silently rewriting the committed bytes from it would destroy the replication
    artifact. If the smaller set is genuinely intended -- Google retired a panorama --
    delete the committed file first and rebuild.
    """
    if not os.path.exists(path):
        return
    old_count = count_existing(path)
    if new_count < old_count:
        raise RuntimeError(
            f"refusing to shrink {os.path.basename(path)}: {old_count} committed "
            f"records but only {new_count} rebuilt -- the fetch cache looks partial. "
            f"Delete the committed file first if the shrink is intended."
        )


def _count_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return sum(1 for _ in f)


def write_tiles_artifact(sample, cache_dir, data_dir, out_dir):
    """Verbatim tile bytes -> data/depth-validation-tiles.jsonl.gz.

    This file IS the replication guarantee: it holds the JPEGs exactly as Google served
    them, so every image-derived number in the report can be recomputed by anyone with
    the repo and no network access at all.
    """
    lines = []
    scoring = set(sample["scoring_a"] + sample["scoring_b"])
    for pano_ids, zoom, set_name in (
        (sorted(scoring), SCORING_ZOOM, "scoring"),
        (sorted(sample["adjudication"]), ADJUDICATION_ZOOM, "adjudication"),
    ):
        for pid in pano_ids:
            resp = load_photometa(pid, cache_dir, data_dir)
            pano = parse_meta(resp) if resp else None
            if pano is None or not pano.image_sizes:
                continue
            width, height, cols, rows, tw, th = tile_grid(pano, zoom)
            tiles = []
            ok = True
            for x in range(cols):
                for y in range(rows):
                    path = os.path.join(cache_dir, "tiles", f"{pid}_{zoom}_{x}_{y}.jpg")
                    if not os.path.exists(path):
                        ok = False
                        break
                    with open(path, "rb") as f:
                        tiles.append({
                            "x": x, "y": y,
                            "b64": base64.b64encode(f.read()).decode("ascii"),
                        })
                if not ok:
                    break
            if not ok:
                continue
            lines.append({
                "pano_id": pid, "set": set_name, "zoom": zoom,
                "width": width, "height": height,
                "tile_width": tw, "tile_height": th,
                "capture_year": pano.date.year if pano.date else None,
                "tiles": tiles,
            })
    lines.sort(key=lambda r: (r["set"], r["pano_id"]))
    out_path = os.path.join(out_dir, "depth-validation-tiles.jsonl.gz")
    _guard_artifact_shrink(out_path, len(lines), _count_jsonl_gz)
    write_gz_bytes(
        out_path,
        "".join(json.dumps(r) + "\n" for r in lines).encode("utf-8"),
    )
    return len(lines)


def write_panometa_artifact(sample, cache_dir, data_dir, out_dir):
    """Per-panorama yaw and road-link bearings -> depth-validation-panometa.csv.gz.

    Small, but it is what makes the coordinate-frame checks replayable offline: the
    yaw is needed to place a label on the raster at all, and the link bearings are the
    external reference that says which frame the raster is in (a link points down the
    street, so the correct convention puts it on road).
    """
    # Every panorama that served depth, not just the imagery sample: the payload-only
    # frame check needs link bearings and has no reason to be limited to 60 panoramas.
    wanted = (set(sample["scoring_a"]) | set(sample["scoring_b"])
              | set(sample["adjudication"]) | set(sample["cross_vintage"]))
    payload_path = os.path.join(data_dir, "depth-pilot-payloads.jsonl.gz")
    if os.path.exists(payload_path):
        with gzip.open(payload_path, "rt", encoding="utf-8") as f:
            wanted |= {json.loads(line)["pano_id"] for line in f}

    rows = []
    for pid in sorted(wanted):
        resp = load_photometa(pid, cache_dir, data_dir)
        pano = parse_meta(resp) if resp else None
        if pano is None or pano.heading is None:
            continue
        links = ";".join(
            f"{math.degrees(link.direction) % 360.0:.3f}" for link in (pano.links or [])
        )
        rows.append({
            "pano_id": pid,
            "yaw_deg": round(math.degrees(pano.heading) % 360.0, 4),
            "pitch_deg": round(math.degrees(pano.pitch), 4) if pano.pitch is not None else None,
            "roll_deg": round(math.degrees(pano.roll), 4) if pano.roll is not None else None,
            "link_bearings_deg": links,
        })
    df = pd.DataFrame(rows).sort_values("pano_id").reset_index(drop=True)
    out_path = os.path.join(out_dir, "depth-validation-panometa.csv.gz")
    _guard_artifact_shrink(out_path, len(df), lambda p: len(pd.read_csv(p)))
    write_csv_gz(df, out_path)
    return len(df)


def write_partners_artifact(partners, cache_dir, data_dir, out_dir):
    """Historical-capture depth payloads -> data/depth-validation-partners.jsonl.gz."""
    lines = []
    for pid, partner_id in sorted(partners.items()):
        resp = load_photometa(partner_id, cache_dir, data_dir)
        if resp is None:
            continue
        pano = parse_meta(resp)
        b64 = gd.extract_depth_b64(resp)
        if pano is None or b64 is None:
            continue
        lines.append({
            "pano_id": partner_id, "partner_of": pid,
            "lat": pano.lat, "lng": pano.lon,
            "heading_deg": math.degrees(pano.heading) if pano.heading is not None else None,
            "elevation_m": pano.elevation,
            "capture_year": pano.date.year if pano.date else None,
            "depth_b64": b64,
        })
    lines.sort(key=lambda r: (r["partner_of"], r["pano_id"]))
    out_path = os.path.join(out_dir, "depth-validation-partners.jsonl.gz")
    _guard_artifact_shrink(out_path, len(lines), _count_jsonl_gz)
    write_gz_bytes(
        out_path,
        "".join(json.dumps(r) + "\n" for r in lines).encode("utf-8"),
    )
    return len(lines)


def load_tiles_artifact(data_dir):
    """pano_id -> {set: record} from the committed tiles file. No network, no cache."""
    path = os.path.join(data_dir, "depth-validation-tiles.jsonl.gz")
    out = {}
    if not os.path.exists(path):
        return out
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out.setdefault(rec["pano_id"], {})[rec["set"]] = rec
    return out


def rgb_from_record(record):
    """Stitch one committed tiles record into an equirectangular RGB array."""
    tiles = [
        {"x": t["x"], "y": t["y"], "bytes": base64.b64decode(t["b64"])}
        for t in record["tiles"]
    ]
    return dv.stitch_tiles(
        tiles, record["width"], record["height"],
        record.get("tile_width"), record.get("tile_height"),
    )


def load_payloads(data_dir):
    """pano_id -> decoded depth payload, from the committed #4 payload file."""
    out = {}
    with gzip.open(
        os.path.join(data_dir, "depth-pilot-payloads.jsonl.gz"), "rt", encoding="utf-8"
    ) as f:
        for line in f:
            rec = json.loads(line)
            out[rec["pano_id"]] = rec["depth_b64"]
    return out


def load_partners(data_dir):
    path = os.path.join(data_dir, "depth-validation-partners.jsonl.gz")
    out = []
    if not os.path.exists(path):
        return out
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------- T4 geometry

def camera_offset_m(lat_a, lng_a, lat_b, lng_b):
    """A's camera relative to B's, in metres east/north. Flat earth is ample at ~10 m."""
    mean_lat = math.radians((lat_a + lat_b) / 2.0)
    return (
        (lng_a - lng_b) * gd.METERS_PER_DEGREE * math.cos(mean_lat),
        (lat_a - lat_b) * gd.METERS_PER_DEGREE,
    )


def match_facade_planes(payload_a, payload_b, dx, dy, dz, min_pixels=200, cos_tol=0.99):
    """Compare the two captures' facade planes as planes, in a shared world frame.

    The ray-projection residual is the wrong instrument for facades: sampled walls sit
    tens of metres out while the two cameras stand metres apart, so a ray cast from one
    capture lands on a different part of the scene in the other and the residual
    measures parallax, not disagreement. Comparing the plane parameters sidesteps
    correspondence entirely.

    A plane is ``n . x + d = 0`` in its own camera's frame (the sign follows
    ``compute_point_cloud``: ground rays give ``n . x = -d``). Re-expressing B's planes
    in A's frame shifts the offset by ``n_B . delta`` and leaves the normal alone, so
    two captures that inherited the same building footprint should return the same
    ``(n, d)`` pair. Returns the perpendicular separation, in metres, of each
    well-supported facade in A from its closest match in B.
    """
    delta = np.array([dx, dy, dz], dtype=np.float64)

    def facades(payload):
        counts = dv.plane_pixel_counts(payload)
        tilt = dv.plane_tilt_deg(payload)
        keep = (counts >= min_pixels) & (tilt >= dv.VERTICAL_TILT_DEG)
        return (
            payload.planes_n[keep].astype(np.float64),
            payload.planes_d[keep].astype(np.float64),
        )

    n_a, d_a = facades(payload_a)
    n_b, d_b = facades(payload_b)
    if len(n_a) == 0 or len(n_b) == 0:
        return np.array([])

    d_b_in_a = d_b + n_b @ delta  # B's planes, expressed in A's frame
    out = []
    for normal, offset in zip(n_a, d_a):
        cos = n_b @ normal
        # A plane and its opposite-facing twin are the same plane.
        aligned = np.abs(cos) >= cos_tol
        if not aligned.any():
            continue
        sign = np.sign(cos[aligned])
        separation = np.abs(offset - sign * d_b_in_a[aligned])
        out.append(float(separation.min()))
    return np.array(out)


def cross_vintage_residuals(payload_a, lat_a, lng_a, payload_b, lat_b, lng_b, stride):
    """Project pano A's modelled GROUND into pano B's frame and compare ranges.

    Both clouds are levelled on their own ground plane first, so the comparison is of
    horizontal geometry and does not depend on knowing either camera's absolute
    elevation. What survives is the thing that matters for a label: do two captures of
    the same street, years apart, put the ground in the same world position?

    Restricted to source points inside ``CROSS_VINTAGE_MAX_RANGE_M``. Ray projection
    assumes the ray meets the same surface in both captures, and with cameras metres
    apart that assumption fails fast with distance -- at 50 m a 12 m baseline lands the
    ray somewhere else entirely, and the residual then measures parallax rather than
    disagreement. Facades are handled by ``match_facade_planes`` instead, which needs
    no correspondence at all.
    """
    h, w = payload_a.height, payload_a.width
    cloud_a = gd.compute_point_cloud(payload_a).reshape(h, w, 3).astype(np.float64)
    idx_a = payload_a.indices.reshape(h, w)

    qc_a = gd.camera_height_qc(payload_a)
    qc_b = gd.camera_height_qc(payload_b)
    if not (math.isfinite(qc_a.ground_height) and math.isfinite(qc_b.ground_height)):
        return np.array([])

    dx_cam, dy_cam = camera_offset_m(lat_a, lng_a, lat_b, lng_b)

    sub = (slice(h // 2, h, stride), slice(0, w, stride))  # below-horizon surfaces only
    pts = cloud_a[sub].reshape(-1, 3)
    src_plane = idx_a[sub].reshape(-1)
    keep = (src_plane != 0) & np.isfinite(pts).all(axis=1)
    pts, src_plane = pts[keep], src_plane[keep]
    if pts.size == 0:
        return np.array([])

    src_tilt = dv.plane_tilt_deg(payload_a)[src_plane]
    near = (np.linalg.norm(pts, axis=1) < CROSS_VINTAGE_MAX_RANGE_M) & (
        src_tilt <= dv.HORIZONTAL_TILT_DEG
    )
    pts = pts[near]
    if pts.size == 0:
        return np.array([])

    # Into B's frame: translate horizontally, and re-level z on B's ground plane.
    px = pts[:, 0] + dx_cam
    py = pts[:, 1] + dy_cam
    pz = pts[:, 2] + (qc_b.ground_height - qc_a.ground_height)

    rng_expected = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    # Direction -> payload pixel, inverting compute_point_cloud's angle convention.
    phi = np.arccos(np.clip(pz / rng_expected, -1.0, 1.0))
    theta = np.arctan2(py, px)
    row = (payload_b.height * (1.0 - phi / np.pi) - 0.5)
    col = (payload_b.width * (((np.pi / 2.0 - theta) / (2.0 * np.pi)) % 1.0) - 0.5)
    row = np.clip(np.round(row).astype(int), 0, payload_b.height - 1)
    col = np.clip(np.round(col).astype(int), 0, payload_b.width - 1)

    t_b = gd.compute_depth_t(payload_b)
    rng_b = t_b[row, col]
    ok = np.isfinite(rng_b) & (rng_b < dv.FLAT_EARTH_MAX_RANGE_M)
    return np.abs(rng_expected[ok] - rng_b[ok])


# ---------------------------------------------------------------------------- build

def cmd_build(args):
    # The committed sample first, the fetch cache second, a fresh draw only as a last
    # resort. choose_samples() strata come from the #4 pano table's pano_class, so any
    # correction upstream redraws it -- and the committed tiles only cover the panos that
    # were actually fetched. Recomputing would silently score a different, tile-less set.
    sample = load_committed_sample(args.data_dir)
    if sample is None:
        sample_path = os.path.join(args.cache_dir, "sample.json")
        if os.path.exists(sample_path):
            with open(sample_path, encoding="utf-8") as f:
                sample = json.load(f)
        else:
            sample = choose_samples(args.data_dir)

    partners_path = os.path.join(args.cache_dir, "partners.json")
    if os.path.exists(partners_path):
        with open(partners_path, encoding="utf-8") as f:
            partners = json.load(f)
    else:
        partners = {}

    # Refresh the committed byte artifacts only when the cache is present; otherwise
    # keep what is committed, which is exactly what makes `build` work offline.
    if os.path.isdir(os.path.join(args.cache_dir, "tiles")):
        n = write_tiles_artifact(sample, args.cache_dir, args.data_dir, args.out_dir)
        print(f"tiles artifact: {n} panorama-zoom records")
        n = write_panometa_artifact(sample, args.cache_dir, args.data_dir, args.out_dir)
        print(f"panometa artifact: {n} panoramas (yaw + link bearings)")
    if partners:
        n = write_partners_artifact(partners, args.cache_dir, args.data_dir, args.out_dir)
        print(f"partners artifact: {n} historical payloads")

    tiles = load_tiles_artifact(args.data_dir)
    payload_b64 = load_payloads(args.data_dir)
    pilot_panos = pd.read_csv(os.path.join(args.data_dir, "depth-pilot-panos.csv.gz"))
    pilot_labels = pd.read_csv(os.path.join(args.data_dir, "depth-pilot-labels.csv.gz"))
    heights = (
        pilot_panos.dropna(subset=["ground_height_m"])
        .groupby("pano_id")["ground_height_m"].first().to_dict()
    )
    coords = (
        pilot_panos.dropna(subset=["fresh_lat", "fresh_lng"])
        .groupby("pano_id")[["fresh_lat", "fresh_lng"]].first()
    )
    meta_rows = pilot_panos.dropna(subset=["pano_id"]).groupby("pano_id").first()

    scoring_ids = sorted(set(tiles) & set(payload_b64))

    # Stitch every scoring panorama once and keep only its sky mask: the permutation
    # null scores each payload against many panoramas' imagery, and re-segmenting per
    # pairing would be the whole runtime.
    sky_masks = {}
    for pid in scoring_ids:
        if "scoring" in tiles[pid]:
            sky_masks[pid] = dv.sky_mask_from_rgb(rgb_from_record(tiles[pid]["scoring"]))
    mask_ids = sorted(sky_masks)
    rng = np.random.default_rng(SEED)
    null_partners = {
        pid: [p for p in rng.permutation(mask_ids) if p != pid][:PERMUTATION_K]
        for pid in mask_ids
    }

    # ---- T1 + T2 per panorama
    pano_records, sweeps = [], {}
    for pid in scoring_ids:
        payload = gd.decode_depth_payload(payload_b64[pid])
        record = {
            "pano_id": pid,
            "city": meta_rows.loc[pid, "city"] if pid in meta_rows.index else None,
            "part": meta_rows.loc[pid, "part"] if pid in meta_rows.index else None,
            "pano_class": meta_rows.loc[pid, "pano_class"] if pid in meta_rows.index else None,
            "capture_year": meta_rows.loc[pid, "capture_year"] if pid in meta_rows.index else None,
        }

        inv = dv.plane_inventory(payload)
        record.update({f"inv_{k}": v for k, v in inv.__dict__.items()})
        height = heights.get(pid, float("nan"))
        record["ground_height_m"] = height
        if math.isfinite(height):
            fe = dv.flat_earth_comparison(payload, height)
            record.update({f"flat_{k}": v for k, v in fe.__dict__.items()})

        record["structure_fraction"] = dv.structure_fraction(payload)
        record["has_power"] = record["structure_fraction"] >= STRUCTURE_MIN
        record["has_imagery"] = pid in sky_masks
        if record["has_imagery"]:
            rec = tiles[pid]["scoring"]
            rgb = rgb_from_record(rec)
            record["image_width"] = rec["width"]
            record["image_height"] = rec["height"]
            for score in dv.registration_scores(payload, rgb):
                record[f"viol_{score.control}"] = score.sky_violation
                record[f"iou_{score.control}"] = score.sky_iou

            # Permutation null: this payload against other panoramas' imagery.
            null = [
                dv.violation_against(payload, sky_masks[p]) for p in null_partners[pid]
            ]
            null = [v for v in null if np.isfinite(v)]
            own = record.get("viol_identity", float("nan"))
            if null and np.isfinite(own):
                record["viol_null_median"] = float(np.median(null))
                record["viol_null_min"] = float(np.min(null))
                record["null_n"] = len(null)
                # Rank of the true pairing among {true} + null: 0 means the payload
                # fits its own panorama better than every mismatched one.
                record["null_rank"] = int(sum(v < own for v in null))
                record["null_percentile"] = float(record["null_rank"] / (len(null) + 1))

            shifts, scores = dv.column_offset_sweep(payload, rgb, max_shift=128, step=4)
            finite = np.isfinite(scores)
            if finite.any():
                record["sweep_argmin_cols"] = int(shifts[finite][np.argmin(scores[finite])])
            sweeps[pid] = (shifts.tolist(), [None if not np.isfinite(s) else float(s) for s in scores])
        pano_records.append(record)

    # ---- T3 per label
    label_records = []
    for pid in scoring_ids:
        rows = pilot_labels[pilot_labels["pano_id"] == pid]
        if rows.empty:
            continue
        payload = gd.decode_depth_payload(payload_b64[pid])
        geom = dv.payload_geometry(payload)  # shared by every label on this panorama
        height = heights.get(pid, float("nan"))
        for _, r in rows.iterrows():
            hit = dv.classify_label_hit(
                payload, int(r["sv_image_x"]), int(r["sv_image_y"]), height,
                geometry=geom,
            )
            label_records.append({
                "pano_id": pid, "label_id": int(r["label_id"]), "city": r["city"],
                "label_type": r["label_type"], "zoom": int(r["zoom"]),
                "sv_image_x": int(r["sv_image_x"]), "sv_image_y": int(r["sv_image_y"]),
                "in_cleaned": bool(r["in_cleaned"]),
                "hit_class": hit.hit_class,
                "range_m": hit.range_m,
                "horizontal_m": hit.horizontal_m,
                "height_above_ground_m": hit.height_above_ground_m,
                "flat_earth_m": hit.flat_earth_m,
                "flat_earth_excess_m": hit.flat_earth_excess_m,
                "neighbourhood_range_ratio": hit.neighbourhood_range_ratio,
                "curb_bias_m": dv.curb_height_bias_m(hit.horizontal_m, height),
                "has_adjudication_imagery": "adjudication" in tiles.get(pid, {}),
            })

    # ---- T4 cross-vintage
    cross_records = []
    for partner in load_partners(args.data_dir):
        pid = partner["partner_of"]
        if pid not in payload_b64 or pid not in coords.index:
            continue
        pa = gd.decode_depth_payload(payload_b64[pid])
        pb = gd.decode_depth_payload(partner["depth_b64"])
        resid = cross_vintage_residuals(
            pa, float(coords.loc[pid, "fresh_lat"]), float(coords.loc[pid, "fresh_lng"]),
            pb, float(partner["lat"]), float(partner["lng"]),
            CROSS_VINTAGE_SAMPLE_STRIDE,
        )
        if resid.size == 0:
            continue
        year_a = meta_rows.loc[pid, "capture_year"] if pid in meta_rows.index else None
        dx, dy = camera_offset_m(
            float(coords.loc[pid, "fresh_lat"]), float(coords.loc[pid, "fresh_lng"]),
            float(partner["lat"]), float(partner["lng"]),
        )
        qc_a, qc_b = gd.camera_height_qc(pa), gd.camera_height_qc(pb)
        facades = match_facade_planes(
            pa, pb, dx, dy, qc_b.ground_height - qc_a.ground_height
        )
        record = {
            "pano_id": pid, "partner_id": partner["pano_id"],
            "capture_year": year_a, "partner_year": partner["capture_year"],
            "year_gap": (
                abs(int(partner["capture_year"]) - int(year_a))
                if partner["capture_year"] and year_a and pd.notna(year_a) else None
            ),
            "separation_m": float(haversine_m(
                float(coords.loc[pid, "fresh_lng"]), float(coords.loc[pid, "fresh_lat"]),
                float(partner["lng"]), float(partner["lat"]),
            )),
            "n_points": int(resid.size),
            "median_residual_m": float(np.median(resid)),
            "p90_residual_m": float(np.percentile(resid, 90)),
            "frac_within_1m": float((resid < 1.0).mean()),
        }
        if facades.size:
            record["n_facades"] = int(facades.size)
            record["median_facade_offset_m"] = float(np.median(facades))
            record["frac_facades_within_1m"] = float((facades < 1.0).mean())
        cross_records.append(record)

    panos_df = pd.DataFrame(pano_records).sort_values("pano_id").reset_index(drop=True)
    labels_df = pd.DataFrame(label_records)
    if not labels_df.empty:
        labels_df = labels_df.sort_values(["pano_id", "label_id"]).reset_index(drop=True)
    cross_df = pd.DataFrame(cross_records)
    if not cross_df.empty:
        cross_df = cross_df.sort_values("pano_id").reset_index(drop=True)

    out = args.out_dir
    write_csv_gz(panos_df, os.path.join(out, "depth-validation-panos.csv.gz"))
    write_csv_gz(labels_df, os.path.join(out, "depth-validation-labels.csv.gz"))
    write_csv_gz(cross_df, os.path.join(out, "depth-validation-crossvintage.csv.gz"))
    write_gz_bytes(
        os.path.join(out, "depth-validation-sweeps.json.gz"),
        json.dumps(sweeps, sort_keys=True).encode("utf-8"),
    )

    summary = summarize(panos_df, labels_df, cross_df, args.data_dir, sweeps)
    with open(
        os.path.join(out, "depth-validation-summary.json"), "w", encoding="utf-8", newline="\n"
    ) as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote depth-validation-* to {out}")


def summarize(panos, labels, cross, data_dir, sweeps=None):
    """Every number the report claims, computed once, here."""
    scored = panos[panos.get("has_imagery", pd.Series(dtype=bool)).fillna(False)]
    # Only panoramas whose model puts something above the horizon can testify about
    # registration; the rest are bare ground under an empty sky, which every frame
    # convention reproduces equally well. Reporting them as failures would be wrong.
    cohort = scored[scored["has_power"].fillna(False)]

    def med(frame, col):
        if col not in frame or frame[col].dropna().empty:
            return None
        return round(float(frame[col].median()), 4)

    def wins(frame, rival):
        """Paired sign test of identity against one rival, ignoring exact ties."""
        if f"viol_{rival}" not in frame:
            return None
        pair = frame[["viol_identity", f"viol_{rival}"]].dropna()
        better = int((pair["viol_identity"] < pair[f"viol_{rival}"]).sum())
        worse = int((pair["viol_identity"] > pair[f"viol_{rival}"]).sum())
        return {"identity_better": better, "rival_better": worse,
                "ties": int(len(pair) - better - worse)}

    controls = list(dv.FRAME_CONTROLS)  # the shuffle control is now the permutation null

    sweep_at_zero = None
    if "sweep_argmin_cols" in cohort:
        vals = cohort["sweep_argmin_cols"].dropna()
        sweep_at_zero = f"{int((vals == 0).sum())}/{int(len(vals))}"

    # The pooled sweep is the statistic that matters: one panorama's curve is noisy,
    # but averaging the cohort's curves shows whether the alignment optimum really
    # sits at zero offset.
    pooled = None
    if sweeps:
        # Only panoramas whose sweep produced any finite score: an empty sky mask
        # yields an all-NaN curve, which would inflate n while pooling nothing.
        ids = [
            p for p in cohort["pano_id"]
            if p in sweeps and any(v is not None for v in sweeps[p][1])
        ]
        if ids:
            shifts = np.array(sweeps[ids[0]][0], dtype=float)
            stack = np.array([
                [np.nan if v is None else v for v in sweeps[p][1]] for p in ids
            ], dtype=float)
            mean_curve = np.nanmean(stack, axis=0)
            pooled = {
                "n_panos": len(ids),
                "argmin_cols": int(shifts[int(np.nanargmin(mean_curve))]),
                "mean_violation_at_0": round(float(mean_curve[list(shifts).index(0.0)]), 5),
                "mean_violation_at_plus_64": round(
                    float(mean_curve[list(shifts).index(64.0)]), 5),
                "mean_violation_at_minus_64": round(
                    float(mean_curve[list(shifts).index(-64.0)]), 5),
            }

    # The whole-corpus characterization runs on all 409 committed payloads, not just
    # the imagery subset -- it needs no imagery, so it should not be limited by it.
    tilt = np.zeros(90)
    inv_rows, fe_rows = [], []
    pilot_panos = pd.read_csv(os.path.join(data_dir, "depth-pilot-panos.csv.gz"))
    heights = (
        pilot_panos.dropna(subset=["ground_height_m"])
        .groupby("pano_id")["ground_height_m"].first().to_dict()
    )
    with gzip.open(
        os.path.join(data_dir, "depth-pilot-payloads.jsonl.gz"), "rt", encoding="utf-8"
    ) as f:
        for line in f:
            rec = json.loads(line)
            payload = gd.decode_depth_payload(rec["depth_b64"])
            tilt += dv.tilt_histogram(payload)
            inv_rows.append(dv.plane_inventory(payload).__dict__)
            h = heights.get(rec["pano_id"], float("nan"))
            if math.isfinite(h):
                fe_rows.append(dv.flat_earth_comparison(payload, h).__dict__)
    tilt = tilt / tilt.sum()
    inv = pd.DataFrame(inv_rows)
    fe = pd.DataFrame(fe_rows)

    lab = labels[labels["in_cleaned"]] if "in_cleaned" in labels else labels
    hit_counts = lab["hit_class"].value_counts().to_dict() if not lab.empty else {}

    # Occlusion is invisible to geometry -- an unmodelled car returns exactly the
    # ground range the flat-earth prediction already gives -- so the verdicts are a
    # committed, hand-made record rather than something recomputed here.
    adjudication = None
    adj_path = os.path.join(data_dir, "depth-validation-adjudication.json")
    if os.path.exists(adj_path):
        with open(adj_path, encoding="utf-8") as f:
            adj = json.load(f)
        adjudication = {
            "sample_size": adj["sample_size"],
            "occluded": adj["occluded"],
            "panoramas_in_sample": adj["panoramas_in_sample"],
            "panoramas_with_an_occlusion": adj["panoramas_with_an_occlusion"],
        }

    return {
        "t1_registration": {
            "panos_with_imagery": int(len(scored)),
            "panos_with_structure": int(len(cohort)),
            "structure_threshold": STRUCTURE_MIN,
            "median_sky_violation": {
                c: med(cohort, f"viol_{c}") for c in controls if f"viol_{c}" in cohort
            },
            "median_sky_iou": {
                c: med(cohort, f"iou_{c}") for c in controls if f"iou_{c}" in cohort
            },
            "paired_sign_test": {
                rival: wins(cohort, rival)
                for rival in ("x_mirror", "rotate_180", "row_flip")
            },
            "permutation_null": {
                "k_per_pano": PERMUTATION_K,
                "median_null_violation": med(cohort, "viol_null_median"),
                "panos_beating_every_mismatch": (
                    f"{int((cohort['null_rank'] == 0).sum())}/"
                    f"{int(cohort['null_rank'].notna().sum())}"
                    if "null_rank" in cohort else None
                ),
                "median_null_percentile": med(cohort, "null_percentile"),
            },
            "column_sweep_minimum_at_zero": sweep_at_zero,
            "pooled_column_sweep": pooled,
        },
        "t2_what_it_is": {
            "payloads": int(len(inv)),
            # The histogram's 1-degree bins make this tilt < 10 deg exactly, hence lt.
            "tilt_pixel_share_horizontal_lt10deg": round(float(tilt[:10].sum()), 4),
            "tilt_pixel_share_vertical_ge80deg": round(float(tilt[80:].sum()), 4),
            "tilt_pixel_share_oblique_15to75deg": round(float(tilt[15:75].sum()), 5),
            # n_planes counts the header's plane list, whose entry 0 is the no-plane
            # marker; subtract it so this counts actual modelled planes.
            "median_planes_per_pano": int((inv["n_planes"] - 1).median()),
            "median_px_share_horizontal": round(float(inv["px_share_horizontal"].median()), 3),
            "median_px_share_vertical": round(float(inv["px_share_vertical"].median()), 3),
            "flat_earth_frac_within_1m": round(float(fe["frac_within_1m"].median()), 4),
            "flat_earth_frac_within_2m": round(float(fe["frac_within_2m"].median()), 4),
            "flat_earth_median_residual_m": round(float(fe["median_residual_m"].median()), 4),
            "deviation_attribution": {
                "terrain": round(float(fe["frac_dev_terrain"].mean()), 3),
                "facade": round(float(fe["frac_dev_facade"].mean()), 3),
                "other": round(float(fe["frac_dev_other"].mean()), 3),
            },
        },
        "t3_label_hits": {
            "labels": int(len(lab)),
            "hit_class_counts": {k: int(v) for k, v in sorted(hit_counts.items())},
            "median_horizontal_m": med(lab, "horizontal_m"),
            "median_curb_bias_m": med(lab, "curb_bias_m"),
            "median_abs_flat_earth_excess_m": (
                round(float(lab["flat_earth_excess_m"].abs().median()), 4)
                if "flat_earth_excess_m" in lab and not lab.empty else None
            ),
            "frac_labels_within_1m_of_flat_earth": (
                round(float((lab["flat_earth_excess_m"].abs() < 1.0).mean()), 4)
                if "flat_earth_excess_m" in lab and not lab.empty else None
            ),
            "occlusion_adjudication": adjudication,
        },
        "t4_cross_vintage": {
            "pairs": int(len(cross)),
            "median_year_gap": (
                int(cross["year_gap"].median()) if not cross.empty
                and cross["year_gap"].notna().any() else None
            ),
            "median_separation_m": med(cross, "separation_m"),
            "ground_median_residual_m": med(cross, "median_residual_m"),
            "ground_median_frac_within_1m": med(cross, "frac_within_1m"),
            # Facades are compared plane-to-plane, so no ray correspondence is needed.
            "facade_median_offset_m": med(cross, "median_facade_offset_m"),
            "facade_median_frac_within_1m": med(cross, "frac_facades_within_1m"),
            "pairs_with_facades": (
                int(cross["n_facades"].notna().sum()) if "n_facades" in cross else 0
            ),
        },
    }


# ---------------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "build", "figures", "gallery"):
        sp = sub.add_parser(name)
        sp.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
        sp.add_argument("--out-dir", default=os.path.join(ROOT, "data"))
        sp.add_argument("--cache-dir", default=os.path.join(ROOT, "data", "depth-validation-cache"))
        if name == "fetch":
            sp.add_argument("--rps", type=float, default=4.0)
        if name in ("figures", "gallery"):
            sp.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    args = parser.parse_args()

    if args.cmd == "fetch":
        cmd_fetch(args)
    elif args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "figures":
        import depth_validation_figures as figs

        figs.cmd_figures(args)
    elif args.cmd == "gallery":
        import depth_validation_figures as figs

        figs.cmd_gallery(args)


if __name__ == "__main__":
    main()
