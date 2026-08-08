"""The depth anchor for issue #7: two independent measurements of the same detections.

Why this stage exists
---------------------
Bearing-only triangulation measures ``H_rig - delta``, where ``delta`` is however far
above an object's ground contact the auto-labeler's detection point sits. Bearings cannot
separate the two: a lower camera and a higher click point produce identical geometry.

That confound is what stops the triangulated camera height from being compared directly
against the depth-measured 2.34 m modern rig, which was calibrated on *human* clicks with
a different convention.

This module removes it. For a sample of GSV panoramas that contribute to well-conditioned
sites, it reads Google's depth raster **at the very same detection pixel** that produced
the bearing. Both measurements then carry the identical ``delta``, so it cancels in the
comparison and what remains is a genuine disagreement between two independent measurement
systems:

- ``r_depth``  — Google's modelled ground surface along that ray (the chain #3 calibrated)
- ``r_tri``    — the leave-one-out intersection of other panoramas' bearings (this issue)

Their disagreement bounds the error of both at once, which is exactly what #7's body asks
for. Nothing here is fitted.

Frames
------
The depth payload raster is **heading-centred** — column 0 is bearing ``pano_yaw - 180``
(the [coordinate conventions report](../reports/2026-08-06-depth-coordinate-conventions.md)
§1). A detection is placed by its *absolute bearing*, so the column is

    col_fraction = ((bearing_deg - pano_yaw_deg + 180) mod 360) / 360

and the row is the detection's own ``y_normalized`` (already a fraction of panorama height
measured from the top, with the horizon at 0.5). The pixel then goes through the same
shared ``modern_truth.classify_modern_label`` path the modern-truth close-out used, so
this lookup is bit-for-bit the one already locked by ``tests/test_depth_conventions.py``.

Getting a frame wrong here is silent, so the comparison is also run under three
deliberately wrong frames (x-mirror, 180-rotation, row-flip). The right frame has to win
by a wide margin or the anchor is not trustworthy — and because triangulation supplies an
independent range for every pixel, that check needs no assumption about which frame is
correct going in.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

import depth_validation as dv
import gsv_depth as gd
import modern_truth as mt
import triangulation as tg

DATA_DIR = tg.DATA_DIR
PAYLOADS = DATA_DIR / "triangulation-depth-payloads.jsonl.gz"
PANOS_CSV = DATA_DIR / "triangulation-depth-panos.csv.gz"

#: Panoramas to sample per GSV run. Enough that the median is pinned to a few centimetres
#: while the committed payload bundle stays a few megabytes.
PANOS_PER_RUN = 120

FRAME_CONTROLS = ("identity", "x_mirror", "rotate_180", "row_flip")


# ======================================================================================
# Selection and fetch
# ======================================================================================

def select_panos(runs=None, per_run: int = PANOS_PER_RUN,
                 data_dir: Path = DATA_DIR, seed: int = tg.SEED) -> pd.DataFrame:
    """Panoramas whose detections have a usable leave-one-out range, sampled per GSV run.

    Sampling is on *panoramas* rather than detections because a payload is fetched once
    per panorama and serves every detection on it.
    """
    runs = list(runs or tg.GSV_RUNS)
    rng = np.random.default_rng(seed)
    out = []
    for run in runs:
        fit = tg.fit_noise(run, data_dir)
        f = fit["frame"]
        d = f[tg.usable(f)]
        if d.empty:
            continue
        # prefer panoramas carrying several usable detections: more comparisons per fetch
        counts = d.groupby("pano_id").size().sort_values(ascending=False)
        pool = counts.index.to_numpy()
        take = pool[: min(len(pool), per_run * 4)]
        pick = rng.choice(take, size=min(per_run, len(take)), replace=False)
        out.append(pd.DataFrame({"run": run, "pano_id": pick}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fetch(runs=None, per_run: int = PANOS_PER_RUN, data_dir: Path = DATA_DIR) -> dict:
    """Fetch and commit verbatim depth payloads. The only stage that touches the network."""
    plan = select_panos(runs, per_run, data_dir)
    rows, payloads = [], []
    for i, r in enumerate(plan.itertuples(), 1):
        rec = {"run": r.run, "pano_id": r.pano_id, "status": "ok"}
        try:
            resp = gd.fetch_photometa_raw(r.pano_id)
            b64 = gd.extract_depth_b64(resp)
            meta = gd.extract_pano_meta(resp)
            if not b64 or not meta:
                rec["status"] = "no_depth"
            else:
                payloads.append({"pano_id": r.pano_id, "run": r.run, "depth_b64": b64})
                rec.update({k: meta.get(k) for k in
                            ("lat", "lng", "heading_deg", "pitch_deg", "roll_deg",
                             "capture_year", "capture_month", "image_sizes")})
        except Exception as exc:                      # noqa: BLE001 - recorded, not raised
            rec["status"] = f"error:{type(exc).__name__}"
        rows.append(rec)
        if i % 25 == 0:
            print(f"    fetched {i}/{len(plan)}", flush=True)

    panos = pd.DataFrame(rows)
    panos.to_csv(PANOS_CSV, index=False, compression="gzip")
    with gzip.open(PAYLOADS, "wt", encoding="utf-8", newline="\n") as fh:
        for p in payloads:
            fh.write(json.dumps(p, sort_keys=True) + "\n")
    return {"n_planned": len(plan), "n_payloads": len(payloads),
            "status_counts": panos["status"].value_counts().to_dict()}


def load_payloads(path: Path = PAYLOADS) -> dict[str, str]:
    if not Path(path).exists():
        return {}
    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["pano_id"]] = rec["depth_b64"]
    return out


def load_panos(path: Path = PANOS_CSV) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ======================================================================================
# The comparison
# ======================================================================================

def depth_ranges(run: str, frame: pd.DataFrame, payloads: dict[str, str],
                 panos: pd.DataFrame, control: str = "identity") -> pd.DataFrame:
    """Depth-derived horizontal range at each detection's own pixel, joined to its truth."""
    yaw = panos.set_index("pano_id")["heading_deg"].to_dict()
    d = frame[tg.usable(frame) & frame["pano_id"].isin(payloads)].copy()
    if d.empty:
        return d
    rows = []
    for pano_id, g in d.groupby("pano_id"):
        py = yaw.get(pano_id)
        if py is None or not np.isfinite(py):
            continue
        try:
            payload = gd.decode_depth_payload(payloads[pano_id])
            geom = dv.payload_geometry(payload)
        except Exception:                              # noqa: BLE001 - skip bad payload
            continue
        w, h = int(g["pano_height"].iloc[0]) * 2, int(g["pano_height"].iloc[0])
        for m in g.itertuples():
            # heading-centred column from the detection's absolute bearing
            frac = ((m.bearing_deg - py + 180.0) % 360.0) / 360.0
            hit = mt.classify_modern_label(
                payload, frac * w, m.y_normalized * h, w, h,
                camera_height=tg.COT_CAMERA_HEIGHT, geometry=geom, control=control)
            rows.append({
                "run": run, "site_id": m.site_id, "pano_id": pano_id,
                "dep_deg": m.dep_deg, "r_tri": m.r_tri, "range_m": m.range_m,
                "hit_class": hit.hit_class,
                "r_depth": hit.horizontal_m,
                "height_above_ground_m": hit.height_above_ground_m,
            })
    return pd.DataFrame(rows)


def anchor(data_dir: Path = DATA_DIR, runs=None) -> dict:
    """Compare the two independent range measurements, and solve for the click offset.

    Because both measurements are taken at the *same* pixel, the detector's click offset
    ``delta`` is common to them and cancels in their ratio. What the ratio measures is
    therefore the disagreement between Google's modelled ground surface and the bearings'
    intersection — with no free parameters anywhere.
    """
    payloads = load_payloads()
    panos = load_panos()
    if not payloads or panos.empty:
        return {"available": False,
                "note": "run `python python/run_triangulation.py fetch` first"}

    runs = list(runs or tg.GSV_RUNS)
    frames, per_run, pooled = {}, {}, []
    for run in runs:
        if run not in set(panos["run"]):
            continue
        fit = tg.fit_noise(run, data_dir)
        frames[run] = fit["frame"]
        cmp_ = depth_ranges(run, fit["frame"], payloads, panos)
        if cmp_.empty:
            continue
        g = cmp_[(cmp_["hit_class"].isin(("ground", "terrain")))
                 & np.isfinite(cmp_["r_depth"]) & (cmp_["r_depth"] > 1.0)]
        if len(g) < 30:
            continue
        pooled.append(g)
        per_run[run] = _pair_stats(g)

    if not pooled:
        return {"available": False, "note": "no comparable detections"}
    allg = pd.concat(pooled, ignore_index=True)

    # frame controls: the right lookup must beat every wrong one, decisively
    controls = {}
    for control in FRAME_CONTROLS:
        parts = [depth_ranges(r, frames[r], payloads, panos, control) for r in frames]
        parts = [p for p in parts if not p.empty]
        if not parts:
            continue
        c = pd.concat(parts, ignore_index=True)
        c = c[(c["hit_class"].isin(("ground", "terrain"))) & np.isfinite(c["r_depth"])
              & (c["r_depth"] > 1.0)]
        if len(c) < 30:
            controls[control] = {"n": int(len(c))}
            continue
        controls[control] = {
            "n": int(len(c)),
            "median_abs_disagreement_m": round(
                float(np.median(np.abs(c["r_depth"] - c["r_tri"]))), 4),
        }

    stats = _pair_stats(allg)
    # delta: implied height from bearings vs implied height from depth, same pixels
    h_tri = float(np.median(allg["r_tri"] * np.tan(np.radians(allg["dep_deg"]))))
    h_depth = float(np.median(allg["r_depth"] * np.tan(np.radians(allg["dep_deg"]))))
    return {
        "available": True,
        "n_panos": int(allg["pano_id"].nunique()),
        "n_detections": int(len(allg)),
        "runs": per_run,
        "pooled": stats,
        "implied_height_from_bearings_m": round(h_tri, 4),
        "implied_height_from_depth_m": round(h_depth, 4),
        "height_gap_m": round(h_tri - h_depth, 4),
        "interpretation": (
            "Both columns are H_rig - delta for the SAME detection pixels, so the "
            "detector's click offset cancels: the gap is a disagreement between the two "
            "measurement systems, not a convention difference."),
        "frame_controls": controls,
    }


def _pair_stats(g: pd.DataFrame) -> dict:
    diff = (g["r_depth"] - g["r_tri"]).to_numpy()
    ratio = (g["r_tri"] / g["r_depth"]).to_numpy()
    return {
        "n": int(len(g)),
        "median_r_depth_m": round(float(np.median(g["r_depth"])), 4),
        "median_r_tri_m": round(float(np.median(g["r_tri"])), 4),
        "median_signed_diff_m": round(float(np.median(diff)), 4),
        "median_abs_diff_m": round(float(np.median(np.abs(diff))), 4),
        "p90_abs_diff_m": round(float(np.percentile(np.abs(diff), 90)), 4),
        "median_ratio_tri_over_depth": round(float(np.median(ratio)), 4),
    }
