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
import mapillary_falsification as mf
import modern_truth as mt
import triangulation as tg

DATA_DIR = tg.DATA_DIR
PAYLOADS = DATA_DIR / "triangulation-depth-payloads.jsonl.gz"
PANOS_CSV = DATA_DIR / "triangulation-depth-panos.csv.gz"


def payloads_path(data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / PAYLOADS.name


def panos_path(data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / PANOS_CSV.name

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
    panos.to_csv(panos_path(data_dir), index=False, compression="gzip")
    with gzip.open(payloads_path(data_dir), "wt", encoding="utf-8", newline="\n") as fh:
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
                # carried through so the quality-gate sweep can tighten on them
                "bearing_resid_deg": m.bearing_resid_deg,
                "sigma_r_m": m.sigma_r_m,
                "n_panos": m.n_panos,
                "hit_class": hit.hit_class,
                "r_depth": hit.horizontal_m,
                "height_above_ground_m": hit.height_above_ground_m,
            })
    return pd.DataFrame(rows)


def _comparable(c: pd.DataFrame) -> pd.DataFrame:
    """The population both systems can score: a ground-family depth hit with a real range."""
    return c[(c["hit_class"].isin(("ground", "terrain"))) & np.isfinite(c["r_depth"])
             & (c["r_depth"] > 1.0)]


def anchor(data_dir: Path = DATA_DIR, runs=None,
           frames: dict[str, pd.DataFrame] | None = None) -> dict:
    """Compare the two independent range measurements, and solve for the click offset.

    Because both measurements are taken at the *same* pixel, the detector's click offset
    ``delta`` is common to them and cancels in their ratio. What the ratio measures is
    therefore the disagreement between Google's modelled ground surface and the bearings'
    intersection — with no free parameters anywhere.

    ``frames`` accepts the noise-fitted member frames the build already holds, keyed by
    run; any run not supplied is fitted here (several minutes each on the larger cities).
    """
    payloads = load_payloads(payloads_path(data_dir))
    panos = load_panos(panos_path(data_dir))
    if not payloads or panos.empty:
        return {"available": False,
                "note": "run `python python/run_triangulation.py fetch` first"}

    runs = list(runs or tg.GSV_RUNS)
    fitted, per_run, pooled = {}, {}, []
    for run in runs:
        if run not in set(panos["run"]):
            continue
        f = (frames or {}).get(run)
        if f is None:
            f = tg.fit_noise(run, data_dir)["frame"]
        fitted[run] = f
        cmp_ = depth_ranges(run, f, payloads, panos)
        if cmp_.empty:
            continue
        g = _comparable(cmp_)
        if len(g) < 30:
            continue
        pooled.append(g)
        per_run[run] = _pair_stats(g)

    if not pooled:
        return {"available": False, "note": "no comparable detections"}
    allg = pd.concat(pooled, ignore_index=True)

    # Frame controls: the right lookup must beat every wrong one, decisively. The identity
    # numbers are read off the population above, and the wrong frames re-run the lookup
    # over exactly the runs that population came from, so all four are scored on one
    # footing (previously the controls pooled every fetched run while the headline gated
    # per run at n >= 30 — identical today, divergent the day a run slips under the gate).
    controls = {"identity": {
        "n": int(len(allg)),
        "median_abs_disagreement_m": round(
            float(np.median(np.abs(allg["r_depth"] - allg["r_tri"]))), 4),
    }}
    for control in FRAME_CONTROLS[1:]:
        parts = [depth_ranges(r, fitted[r], payloads, panos, control) for r in per_run]
        parts = [_comparable(p) for p in parts if not p.empty]
        c = (pd.concat(parts, ignore_index=True) if parts
             else pd.DataFrame(columns=["r_depth", "r_tri"]))
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
        "position_drift": position_drift(panos, data_dir),
        "quality_gates": quality_gates(allg),
        "gap_range_profile": gap_range_profile(allg),
        "gap_by_capture_year": gap_by_capture_year(allg, panos),
    }


def position_drift(panos: pd.DataFrame, data_dir: Path = DATA_DIR) -> dict:
    """Stored auto-labeler panorama positions against the freshly fetched photometa.

    Load-bearing for the whole report: triangulated range scales with the baseline, so if
    Google had re-estimated these camera positions between the auto-labeler's crawl and
    this fetch, every range would silently inherit the drift. Computed and committed here
    (and locked by the findings tests) rather than asserted in prose.
    """
    per_run, alld = {}, []
    for run, g in panos.groupby("run"):
        g = g[np.isfinite(g["lat"]) & np.isfinite(g["lng"])]
        stored = mf.load_panos(run, data_dir)[["pano_id", "lat", "lng"]]
        m = g.merge(stored, on="pano_id", suffixes=("_fresh", "_stored"))
        if m.empty:
            continue
        lat0, lng0 = float(m["lat_stored"].mean()), float(m["lng_stored"].mean())
        e1, n1 = tg.local_en(m["lat_fresh"], m["lng_fresh"], lat0, lng0)
        e0, n0 = tg.local_en(m["lat_stored"], m["lng_stored"], lat0, lng0)
        d = np.hypot(e1 - e0, n1 - n0)
        alld.append(d)
        per_run[run] = {"n": int(len(m)), "median_m": round(float(np.median(d)), 4),
                        "max_m": round(float(np.max(d)), 4)}
    if not alld:
        return {}
    d = np.concatenate(alld)
    return {"per_run": per_run, "n": int(len(d)),
            "median_m": round(float(np.median(d)), 4),
            "p99_m": round(float(np.percentile(d, 99)), 4),
            "max_m": round(float(np.max(d)), 4)}


def quality_gates(allg: pd.DataFrame, min_n: int = 40) -> dict:
    """The disagreement under tightening quality gates — computed, not asserted.

    "The gap is systematic" is only a claim if it survives every gate that would remove a
    quality artifact: a mis-clustered member (bearing residual), an ill-conditioned
    triangulation (propagated range sigma), a thin site (panorama count). Each sweep
    tightens one gate over the same pooled population and reports the ratio that remains.
    Committed to the summary and locked by the findings tests so the prose can never again
    carry numbers the build does not produce.
    """
    def cell(g: pd.DataFrame) -> dict:
        return {"n": int(len(g)),
                "median_ratio": round(float(np.median(g["r_tri"] / g["r_depth"])), 4)}

    out = {"by_max_abs_bearing_resid_deg": {}, "by_sigma_r_m": {}, "by_min_panos": {}}
    for t in (4.0, 2.0, 1.0, 0.5, 0.25):
        g = allg[np.abs(allg["bearing_resid_deg"]) <= t]
        if len(g) >= min_n:
            out["by_max_abs_bearing_resid_deg"][str(t)] = cell(g)
    for t in (1.5, 1.0, 0.75, 0.5):
        # 1.5 is the headline's own gate (`usable`), so that row restates the pooled ratio
        g = allg[allg["sigma_r_m"] <= t]
        if len(g) >= min_n:
            out["by_sigma_r_m"][str(t)] = cell(g)
    for t in (3, 4, 5):
        g = allg[allg["n_panos"] >= t]
        if len(g) >= min_n:
            out["by_min_panos"][str(t)] = cell(g)
    return out


#: Range-bin edges for the gap profile. The last committed bin (18, 25] still carries
#: ~300 detections; past 25 m the population is single digits and says nothing.
GAP_RANGE_BINS = (1, 5, 8, 11, 14, 18, 25)


def gap_range_profile(allg: pd.DataFrame, bins=GAP_RANGE_BINS, min_n: int = 40) -> dict:
    """Is the disagreement a constant *ratio* or a constant *offset*? They differ in range.

    The two candidate causes predict different shapes. A depth model whose scale is set by
    its own assumed ground plane is wrong *multiplicatively*: the ratio is flat in range
    and the metre gap grows in proportion. A detector centroid displaced toward the camera
    on a fixed-size object is wrong *additively*: the metre gap is capped by the object's
    extent and the ratio must fall toward 1 as range grows. (A radial displacement also
    leaves every bearing unchanged — the contract tests prove the ray intersection cannot
    see it — so it could only enter this comparison through the depth side's pixel.)
    """
    k = pd.cut(allg["r_tri"], bins=list(bins))
    out = {}
    for key, g in allg.groupby(k, observed=True):
        if len(g) < min_n:
            continue
        out[str(key)] = {
            "n": int(len(g)),
            "median_r_tri_m": round(float(np.median(g["r_tri"])), 3),
            "median_diff_m": round(float(np.median(g["r_tri"] - g["r_depth"])), 4),
            "median_ratio": round(float(np.median(g["r_tri"] / g["r_depth"])), 4),
            "median_h_depth_m": round(float(np.median(
                g["r_depth"] * np.tan(np.radians(g["dep_deg"])))), 4),
        }
    return out


def gap_by_capture_year(allg: pd.DataFrame, panos: pd.DataFrame,
                        edges=(2007, 2016, 2020, 2023, 2027), min_n: int = 40) -> dict:
    """The gap stratified by capture era.

    The modern-truth close-out measured the depth planes' scale to be era-dependent (the
    era fleet's pinned 2.50 m planes). If the same pathology drove this gap, the ratio
    would track capture era; a flat profile across the modern bulk says the headline is
    not an old-imagery artifact.
    """
    e = allg.merge(panos[["pano_id", "capture_year"]], on="pano_id", how="left")
    k = pd.cut(e["capture_year"], bins=list(edges))
    out = {}
    for key, g in e.groupby(k, observed=True):
        if len(g) < min_n:
            continue
        out[str(key)] = {
            "n": int(len(g)),
            "median_ratio": round(float(np.median(g["r_tri"] / g["r_depth"])), 4),
            "median_diff_m": round(float(np.median(g["r_tri"] - g["r_depth"])), 4),
        }
    return out


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
