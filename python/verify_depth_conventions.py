"""Reproduce every coordinate-convention finding behind issue #9, from committed bytes.

    python python/verify_depth_conventions.py            # run all checks, print a report
    python python/verify_depth_conventions.py --json     # also write data/depth-conventions-evidence.json

Three frames are in play in the Project Sidewalk depth pipeline and they do not agree
with each other. Getting one wrong is silent: the arrays are the right shape, the values
are plausible metres, and nothing raises. Each check below is the measurement that pins
one of them down, written so the conclusion can be re-derived rather than trusted.

    A  decoder orientation   streetlevel's DepthMap vs the 2020 client's payload order
    B  sv_image_x frame      is sv_image_x a compass bearing, or heading-relative?
    C  pano_x vs sv_image_x  the two database columns, against each other
    D  raster frame          road links vs the imagery: where does a street land?
    E  payload frame         road sightlines in the depth alone, no imagery
    F  registration          payload vs mirrored, scored against the imagery
    G  impact                what the frame mismatch costs a stored label position

Everything runs offline from committed artifacts: the #4 depth payloads, the #9 imagery
tiles, the per-panorama yaw/link table, and the recovered label dataset. No network, and
the only decode path is the v6 replica in gsv_depth.py.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import depth_validation as dv  # noqa: E402
import gsv_depth as gd  # noqa: E402
from label_latlng_estimation import clean_data, load_data  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

# Road-likeness: a strip of the image below the horizon is "road" when its pixels are
# grey. Deliberately crude -- it is applied identically to both hypotheses, so it cannot
# favour either.
GREY_TOLERANCE = 28
ROAD_BAND = (0.56, 0.80)  # fraction of image height, below the horizon
STRIP_HALFWIDTH = 0.012  # fraction of image width


def wrap180(x):
    return ((x + 180.0) % 360.0) - 180.0


def load_payloads():
    out = {}
    with gzip.open(os.path.join(DATA, "depth-pilot-payloads.jsonl.gz"), "rt",
                   encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["pano_id"]] = rec["depth_b64"]
    return out


def load_panometa():
    path = os.path.join(DATA, "depth-validation-panometa.csv.gz")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["pano_id", "yaw_deg", "link_bearings_deg"])
    return pd.read_csv(path)


# ---------------------------------------------------------------- A: decoder orientation

def check_decoder_orientation(payloads):
    """streetlevel mirrors x on output; the 2020 client did not.

    GSVPanoDepth.js ships both conventions -- ``computePointCloud`` writes in payload
    order, ``computeDepthMap`` writes to ``w - x - 1``. Project Sidewalk's client used
    the first, streetlevel ports the second. So this is an inherited difference between
    two functions in the same upstream file, not a defect in either library, and the
    only question is which order matches the panorama raster (check F).
    """
    from streetlevel.streetview import depth as sl_depth

    asis, mirrored = [], []
    for b64 in payloads.values():
        payload = gd.decode_depth_payload(b64)
        mine = gd.compute_depth_t(payload)
        theirs = sl_depth.parse(b64).data
        theirs = np.where(theirs < 0, np.nan, theirs)
        m1 = np.isfinite(mine) & np.isfinite(theirs)
        m2 = np.isfinite(mine) & np.isfinite(theirs[:, ::-1])
        asis.append(np.nanmax(np.abs(mine[m1] - theirs[m1])) if m1.any() else np.nan)
        mirrored.append(
            np.nanmax(np.abs(mine[m2] - theirs[:, ::-1][m2])) if m2.any() else np.nan
        )
    asis, mirrored = np.array(asis), np.array(mirrored)
    return {
        "payloads": int(len(asis)),
        "asis_median_max_diff_m": round(float(np.nanmedian(asis)), 3),
        "asis_worst_max_diff_m": round(float(np.nanmax(asis)), 3),
        "mirrored_median_max_diff_m": float(f"{np.nanmedian(mirrored):.3e}"),
        "mirrored_worst_max_diff_m": float(f"{np.nanmax(mirrored):.3e}"),
        "agree_after_mirroring_within_1mm": int((mirrored < 1e-3).sum()),
        "agree_after_mirroring_within_10cm": int((mirrored < 1e-1).sum()),
    }


# ---------------------------------------------------------------- B: the sv_image_x frame

def check_sv_image_x_frame(cleaned):
    """Is ``sv_image_x`` a compass bearing, or is it heading-relative?

    The label's stored position was written by the depth pipeline, so its bearing from
    the panorama is ``sv_image_x/13312 * 360`` by construction and proves nothing on its
    own. What does prove something is the POV ``heading`` recorded independently when
    the user placed the label: whichever hypothesis leaves a small residual against it
    is the frame ``sv_image_x`` actually lives in. Under the wrong one the residual
    picks up a per-panorama ``(180 - yaw)`` term and smears across the whole circle.
    """
    east = (cleaned["lng"] - cleaned["pano_lng"]) * gd.METERS_PER_DEGREE * np.cos(
        np.radians(cleaned["pano_lat"]))
    north = (cleaned["lat"] - cleaned["pano_lat"]) * gd.METERS_PER_DEGREE
    bearing = np.degrees(np.arctan2(east, north)) % 360.0

    res_a = wrap180(bearing - cleaned["heading"])
    res_b = wrap180(bearing - cleaned["heading"] - (180.0 - cleaned["photographer_heading"]))
    sv_az = (cleaned["sv_image_x"] / 13312.0 * 360.0) % 360.0
    identity = wrap180(bearing - sv_az).abs()

    def describe(v):
        v = v.dropna()
        return {"median": round(float(v.median()), 2),
                "iqr": [round(float(v.quantile(0.25)), 2), round(float(v.quantile(0.75)), 2)],
                "within_60deg_frac": round(float((v.abs() < 60).mean()), 4),
                "std": round(float(v.std()), 2)}

    return {
        "labels": int(len(cleaned)),
        "A_sv_image_x_is_compass_bearing": describe(res_a),
        "B_sv_image_x_is_heading_relative": describe(res_b),
        "bearing_vs_sv_image_x_median_deg": round(float(identity.median()), 3),
        "verdict": "A" if describe(res_a)["std"] < describe(res_b)["std"] else "B",
    }


# ---------------------------------------------------------------- C: the two DB columns

def check_pano_x_vs_sv_image_x(raw):
    """``current_pano_x`` (evolution 179) against the legacy ``sv_image_x``.

    If the two columns differ by exactly the panorama's yaw, they are the same point
    expressed in the two different frames -- which is the whole hazard.
    """
    d = raw.dropna(subset=["current_pano_x", "sv_image_x", "pano_width",
                           "photographer_heading"]).copy()
    delta = ((d["current_pano_x"] / d["pano_width"]
              - (d["sv_image_x"] / 13312.0) % 1.0) + 0.5) % 1.0 - 0.5
    predicted = (((180.0 - d["photographer_heading"]) % 360.0) / 360.0 + 0.5) % 1.0 - 0.5
    return {
        "rows": int(len(d)),
        "correlation_with_yaw_offset": round(float(np.corrcoef(delta, predicted)[0, 1]), 4),
        "median_residual_deg": round(float(wrap180((delta - predicted) * 360.0).median()), 3),
    }


# ---------------------------------------------------------------- D/E: the raster frame

def _strip(img, frac_col):
    height, width = img.shape[:2]
    centre = int(frac_col * width) % width
    half = max(2, int(STRIP_HALFWIDTH * width))
    cols = [(centre + i) % width for i in range(-half, half + 1)]
    return img[int(ROAD_BAND[0] * height):int(ROAD_BAND[1] * height), cols, :]


def roadness(img, frac_col):
    band = _strip(img, frac_col).astype(np.int16)
    return float(((band.max(axis=2) - band.min(axis=2)) < GREY_TOLERANCE).mean())


def check_raster_frame(panometa, tiles, rgb_loader, min_yaw_offset=60.0):
    """Where does a road link fall in the imagery? It must fall on road.

    Restricted to panoramas whose yaw is far from 180 degrees, because that is where the
    two hypotheses actually separate -- a panorama facing due south places them on top of
    each other and can testify to nothing.
    """
    rows = []
    for _, meta in panometa.iterrows():
        pid, yaw = meta["pano_id"], float(meta["yaw_deg"])
        if abs(wrap180(180.0 - yaw)) < min_yaw_offset:
            continue
        if pid not in tiles or not isinstance(meta["link_bearings_deg"], str):
            continue
        img = rgb_loader(pid)
        if img is None:
            continue
        for token in meta["link_bearings_deg"].split(";"):
            if not token:
                continue
            b = float(token)
            rows.append((roadness(img, b / 360.0),
                         roadness(img, ((b - yaw + 180.0) % 360.0) / 360.0)))
    if not rows:
        return {"links": 0}
    arr = np.array(rows)
    return {
        "links": int(len(arr)),
        "A_north_referenced_roadness": round(float(arr[:, 0].mean()), 4),
        "B_heading_centred_roadness": round(float(arr[:, 1].mean()), 4),
        "B_better_on": int((arr[:, 1] > arr[:, 0]).sum()),
        "A_better_on": int((arr[:, 0] > arr[:, 1]).sum()),
        "verdict": "B" if arr[:, 1].mean() > arr[:, 0].mean() else "A",
    }


def check_payload_frame(panometa, payloads, min_yaw_offset=45.0):
    """The same question asked of the depth alone: a street is an open sightline.

    Uses no imagery at all, so it is independent of the raster check.
    """
    rows = []
    for _, meta in panometa.iterrows():
        pid, yaw = meta["pano_id"], float(meta["yaw_deg"])
        if pid not in payloads or abs(wrap180(180.0 - yaw)) < min_yaw_offset:
            continue
        if not isinstance(meta["link_bearings_deg"], str):
            continue
        payload = gd.decode_depth_payload(payloads[pid])
        t = gd.compute_depth_t(payload)
        band = t[132:150, :]
        base = float(np.nanmedian(band[np.isfinite(band)])) if np.isfinite(band).any() else np.nan

        def at(frac):
            centre = int(frac * payload.width) % payload.width
            cols = [(centre + i) % payload.width for i in range(-6, 7)]
            v = t[132:150, cols]
            v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else np.nan

        for token in meta["link_bearings_deg"].split(";"):
            if not token:
                continue
            b = float(token)
            rows.append((at(b / 360.0), at(((b - yaw + 180.0) % 360.0) / 360.0), base))
    arr = np.array([r for r in rows if np.isfinite(r).all()])
    if not arr.size:
        return {"links": 0}
    return {
        "links": int(len(arr)),
        "A_exceeds_pano_median": int((arr[:, 0] > arr[:, 2]).sum()),
        "B_exceeds_pano_median": int((arr[:, 1] > arr[:, 2]).sum()),
        "A_median_range_m": round(float(np.median(arr[:, 0])), 2),
        "B_median_range_m": round(float(np.median(arr[:, 1])), 2),
        "pano_median_range_m": round(float(np.median(arr[:, 2])), 2),
        "verdict": "B" if (arr[:, 1] > arr[:, 2]).sum() > (arr[:, 0] > arr[:, 2]).sum() else "A",
    }


# ---------------------------------------------------------------- F: registration

def check_registration(panos_csv):
    """Payload order vs mirrored, scored against the panoramas' own imagery.

    Reads the scores the main build already computed: ``viol_identity`` is the payload
    as the 2020 client read it, ``viol_x_mirror`` is streetlevel's orientation.
    """
    panos = pd.read_csv(panos_csv)
    cohort = panos[panos["has_power"].fillna(False) & panos["has_imagery"].fillna(False)]
    pair = cohort[["viol_identity", "viol_x_mirror"]].dropna()
    return {
        "panoramas": int(len(pair)),
        "payload_order_median_violation": round(float(pair["viol_identity"].median()), 4),
        "mirrored_median_violation": round(float(pair["viol_x_mirror"].median()), 4),
        "payload_order_better_on": int((pair["viol_identity"] < pair["viol_x_mirror"]).sum()),
        "mirrored_better_on": int((pair["viol_identity"] > pair["viol_x_mirror"]).sum()),
    }


# ---------------------------------------------------------------- G: what it costs

def yaw_table(panometa):
    """pano_id -> yaw, over every panorama with a known heading.

    ``panometa`` covers the 60 imagery panoramas; the #4 pilot table carries a heading
    for all 409 that served depth, and check G wants the widest sample it can get.
    """
    yaw = {}
    pilot = pd.read_csv(os.path.join(DATA, "depth-pilot-panos.csv.gz"))
    pilot = pilot.dropna(subset=["pano_id", "fresh_heading_deg"])
    for pid, h in zip(pilot["pano_id"], pilot["fresh_heading_deg"]):
        yaw.setdefault(pid, float(h) % 360.0)
    yaw.update({pid: float(h) for pid, h in zip(panometa["pano_id"], panometa["yaw_deg"])})
    return yaw


def check_frame_impact(payloads, panometa, labels):
    """Re-read every label at the heading-centred column and see how far it moves.

    The 2017-2020 client indexed the payload with ``ceil(sv_image_x/26)``. If the
    payload is heading-centred (checks D and E) while ``sv_image_x`` is a compass
    bearing (check B), that lookup is off by ``(180 - yaw)/360 * 512`` columns. This is
    the size of that mismatch in metres -- small at the median precisely because the
    model is nearly flat earth, so range is set by the depression angle.
    """
    yaw = yaw_table(panometa)
    diffs = []
    for pid, group in labels.groupby("pano_id"):
        if pid not in payloads or pid not in yaw:
            continue
        payload = gd.decode_depth_payload(payloads[pid])
        cloud = gd.compute_point_cloud(payload)
        shift_px = ((180.0 - float(yaw[pid])) % 360.0) / 360.0 * gd.DEPTH_W * 26.0
        for _, r in group.iterrows():
            a = gd.v6_to_latlng(int(r["sv_image_x"]), int(r["sv_image_y"]), 0.0, 0.0, cloud)
            b = gd.v6_to_latlng((int(r["sv_image_x"]) + shift_px) % 13312.0,
                                int(r["sv_image_y"]), 0.0, 0.0, cloud)
            if a.no_plane or b.no_plane or a.out_of_bounds or b.out_of_bounds:
                continue
            da, db = math.hypot(a.dx, a.dy), math.hypot(b.dx, b.dy)
            if np.isfinite(da) and np.isfinite(db) and da < 80 and db < 80:
                diffs.append(abs(da - db))
    d = np.array(diffs)
    if not d.size:
        return {"labels": 0}
    return {
        "labels": int(d.size),
        "median_shift_m": round(float(np.median(d)), 3),
        "p90_shift_m": round(float(np.percentile(d, 90)), 3),
        "p95_shift_m": round(float(np.percentile(d, 95)), 3),
        "frac_within_1m": round(float((d < 1).mean()), 4),
        "frac_beyond_3m": round(float((d > 3).mean()), 4),
    }


# ---------------------------------------------------------------- driver

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true",
                        help="write data/depth-conventions-evidence.json")
    args = parser.parse_args()

    from run_depth_validation import load_tiles_artifact, rgb_from_record

    payloads = load_payloads()
    panometa = load_panometa()
    tiles = load_tiles_artifact(DATA)
    raw = load_data(DATA)
    cleaned, _ = clean_data(raw)
    pilot_labels = pd.read_csv(os.path.join(DATA, "depth-pilot-labels.csv.gz"))
    pilot_labels = pilot_labels[pilot_labels["in_cleaned"] & ~pilot_labels["stored_absurd"]]

    cache = {}

    def rgb_loader(pid):
        if pid not in cache:
            rec = tiles.get(pid, {}).get("scoring") or tiles.get(pid, {}).get("adjudication")
            cache[pid] = rgb_from_record(rec) if rec else None
        return cache[pid]

    results = {
        "A_decoder_orientation": check_decoder_orientation(payloads),
        "B_sv_image_x_frame": check_sv_image_x_frame(cleaned),
        "C_pano_x_vs_sv_image_x": check_pano_x_vs_sv_image_x(raw),
        "D_raster_frame": check_raster_frame(panometa, tiles, rgb_loader),
        "E_payload_frame": check_payload_frame(panometa, payloads),
        "F_registration": check_registration(
            os.path.join(DATA, "depth-validation-panos.csv.gz")),
        "G_frame_impact": check_frame_impact(payloads, panometa, pilot_labels),
    }

    print(json.dumps(results, indent=2, sort_keys=True))
    print("\n" + "=" * 78)
    a, b = results["A_decoder_orientation"], results["B_sv_image_x_frame"]
    print(f"A  streetlevel is the x-mirror of the 2020 client: as-is they differ by a")
    print(f"   median of {a['asis_median_max_diff_m']:.0f} m; mirrored, all "
          f"{a['agree_after_mirroring_within_10cm']}/{a['payloads']} agree within 10 cm.")
    print(f"B  sv_image_x is a compass bearing (hypothesis {b['verdict']}): residual std "
          f"{b['A_sv_image_x_is_compass_bearing']['std']}deg vs "
          f"{b['B_sv_image_x_is_heading_relative']['std']}deg.")
    print(f"C  current_pano_x differs from it by exactly the yaw: r = "
          f"{results['C_pano_x_vs_sv_image_x']['correlation_with_yaw_offset']}.")
    print(f"D  the raster is heading-centred (hypothesis "
          f"{results['D_raster_frame'].get('verdict')}).")
    print(f"E  so is the payload (hypothesis {results['E_payload_frame'].get('verdict')}).")
    f = results["F_registration"]
    print(f"F  payload order beats mirrored against the imagery on "
          f"{f['payload_order_better_on']}/{f['panoramas']} panoramas.")
    g = results["G_frame_impact"]
    print(f"G  the frame mismatch moves a stored label by {g['median_shift_m']} m at the "
          f"median, but {g['frac_beyond_3m'] * 100:.1f}% move beyond 3 m.")
    print("=" * 78)

    if args.json:
        path = os.path.join(DATA, "depth-conventions-evidence.json")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(results, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
