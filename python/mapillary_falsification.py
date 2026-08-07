"""Issue #3 Stage 3 — falsify the refit candidates on Mapillary, don't fit on them.

This module consumes the committed ``data/falsification-*`` inputs (auto-labeler fused
multi-view sites plus per-pano metadata; see ``data/MANIFEST.md``) and currently implements
the opening move: the **projection/metadata census** of the two Mapillary-viewer cities.
The census answers, before any diagnostic runs, whether the falsification's assumptions hold:

- projection — every pano a true 2:1 equirect? (`camera_type`, width == 2·height)
- which pose fields are usable — raw `compass_angle` is 56% literal-zero in clovis, so any
  scoring must use the SfM `computed_*` pose, and the census quantifies raw-vs-computed shifts
- the rig zoo — camera make/model per sequence, pano heights per rig, and a capture-mode
  classification (on-foot / slow / vehicle) per sequence, because camera height above ground
  is rig-dependent and the refit's ``h[t]`` was fit on a ~2.6 m GSV car

Capture-mode is derived, not served: Mapillary has no on-foot flag. Per-sequence we compute a
frame-median speed (median of per-step distance/dt) and a gross speed (total path length over
total duration). Several rigs stamp `captured_at` at fixed or sub-second intervals while
driving (observed: 150 m steps stamped ~1 s apart → absurd 100+ m/s frame speeds), so the
classifier trusts the *smaller* of the two speeds and the frame spacing together.

Everything is deterministic and offline; the runner (``run_mapillary_falsification.py``)
writes ``data/falsification-summary.json``.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

MAPILLARY_RUNS = ["richmond", "clovis"]
GSV_RUNS = ["paterson", "gainesville", "bend", "sao_paulo"]

# Capture-mode thresholds (m/s, m). Grounded in the observed per-sequence distribution:
# clovis walking sequences sit at 0.17-0.85 m/s with sub-meter spacing; confirmed cycling
# (GoPro Max, 3.1 m spacing) at ~4 m/s; car sequences 6.5+ m/s or coarse fixed-distance
# spacing. 2.0 m/s is comfortably above walking pace and below any wheeled mode observed.
ON_FOOT_MAX_SPEED = 2.0
VEHICLE_MIN_SPEED = 5.5
VEHICLE_MIN_SPACING = 25.0
MAX_STEP_SECONDS = 60.0  # steps longer than this are recording gaps, not motion


def haversine_m(lng1, lat1, lng2, lat2):
    """Spherical great-circle meters (matches production turf; see #12 report §1)."""
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    half_dp = (p2 - p1) / 2.0
    half_dl = (np.radians(lng2) - np.radians(lng1)) / 2.0
    a = np.sin(half_dp) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(half_dl) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def load_panos(run: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    # Mapillary pano/sequence/creator ids are all-digit strings; without an explicit dtype
    # pandas parses them as int64 and joins against the sites' string ids silently miss.
    return pd.read_csv(data_dir / f"falsification-panos-{run}.csv.gz",
                       dtype={"pano_id": str, "sequence_id": str, "creator_id": str})


def load_sites(run: str, data_dir: Path = DATA_DIR) -> list[dict]:
    with gzip.open(data_dir / f"falsification-sites-{run}.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def site_member_counts(sites: list[dict]) -> pd.Series:
    """pano_id -> number of fused-site members it contributes (in_refit members only)."""
    counts: dict[str, int] = {}
    for site in sites:
        for member in site["members"]:
            if member.get("in_refit", True):
                counts[member["pano_id"]] = counts.get(member["pano_id"], 0) + 1
    return pd.Series(counts, name="n_members", dtype="int64")


def _pct(x, q) -> float:
    return float(np.percentile(np.asarray(x, dtype=float), q))


def _wrap180(deg):
    return (np.asarray(deg, dtype=float) + 180.0) % 360.0 - 180.0


def sequence_table(panos: pd.DataFrame, members: pd.Series) -> pd.DataFrame:
    """One row per sequence: rig, size, motion estimates, capture-mode, member share."""
    rows = []
    n_members_by_pano = panos["pano_id"].map(members).fillna(0).astype(int)
    panos = panos.assign(_members=n_members_by_pano)
    for sid, g in panos.sort_values(["captured_at_ms", "pano_id"]).groupby("sequence_id"):
        row = {
            "sequence_id": sid,
            "n_panos": len(g),
            "camera_make": g["camera_make"].mode(dropna=False).iloc[0],
            "camera_model": g["camera_model"].mode(dropna=False).iloc[0],
            "camera_type": g["camera_type"].mode(dropna=False).iloc[0],
            "pano_height": int(g["height"].mode().iloc[0]),
            "creator": g["creator_username"].mode(dropna=False).iloc[0],
            "n_detections": int(g["n_detections"].sum()),
            "n_site_members": int(g["_members"].sum()),
            "frame_speed_mps": np.nan,
            "gross_speed_mps": np.nan,
            "spacing_m": np.nan,
        }
        if len(g) >= 3:
            dt = np.diff(g["captured_at_ms"].to_numpy()) / 1000.0
            dd = haversine_m(g["lng"].to_numpy()[:-1], g["lat"].to_numpy()[:-1],
                             g["lng"].to_numpy()[1:], g["lat"].to_numpy()[1:])
            ok = (dt > 0) & (dt <= MAX_STEP_SECONDS)
            if ok.sum() >= 2:
                row["frame_speed_mps"] = float(np.median(dd[ok] / dt[ok]))
                row["spacing_m"] = float(np.median(dd[ok]))
                duration = float(np.sum(dt[ok]))
                if duration > 0:
                    row["gross_speed_mps"] = float(np.sum(dd[ok]) / duration)
        rows.append(row)
    table = pd.DataFrame(rows)
    # Trust the smaller of the two speed estimates: fixed-interval timestamping inflates the
    # frame speed (150 m steps stamped ~1 s apart), while stop-and-go traffic deflates the
    # gross speed — a sequence is only as fast as its slower defensible estimate.
    speed = table[["frame_speed_mps", "gross_speed_mps"]].min(axis=1)
    mode = pd.Series("slow", index=table.index)
    mode[(speed < ON_FOOT_MAX_SPEED) & (table["spacing_m"] < 4.0)] = "on_foot"
    mode[(speed > VEHICLE_MIN_SPEED) | (table["spacing_m"] > VEHICLE_MIN_SPACING)] = "vehicle"
    mode[speed.isna()] = "unknown"
    table["capture_mode"] = mode
    table["speed_mps"] = speed
    return table


def census_mapillary_run(run: str, data_dir: Path = DATA_DIR) -> dict:
    panos = load_panos(run, data_dir)
    members = site_member_counts(load_sites(run, data_dir))
    seq = sequence_table(panos, members)

    rig_key = (panos["camera_make"].fillna("(none)") + " / "
               + panos["camera_model"].fillna("(none)"))
    n_members_by_pano = panos["pano_id"].map(members).fillna(0).astype(int)
    rigs = {}
    for rig, g in panos.assign(_rig=rig_key, _members=n_members_by_pano).groupby("_rig"):
        rigs[rig] = {
            "n_panos": len(g),
            "n_sequences": int(g["sequence_id"].nunique()),
            "n_site_members": int(g["_members"].sum()),
            "pano_dims": {f"{w}x{h}": int(n) for (w, h), n in
                          g.groupby(["width", "height"]).size().items()},
            "camera_type": sorted(g["camera_type"].dropna().unique().tolist()),
        }

    pos_shift = haversine_m(panos["raw_lng"], panos["raw_lat"], panos["lng"], panos["lat"])
    compass_delta = np.abs(_wrap180(panos["computed_compass_angle"] - panos["compass_angle"]))
    alt_delta = panos["computed_altitude"] - panos["altitude"]

    mode_agg = {
        m: {
            "n_sequences": int((seq["capture_mode"] == m).sum()),
            "n_panos": int(seq.loc[seq["capture_mode"] == m, "n_panos"].sum()),
            "n_site_members": int(seq.loc[seq["capture_mode"] == m, "n_site_members"].sum()),
        }
        for m in ["on_foot", "slow", "vehicle", "unknown"]
    }

    return {
        "n_panos": len(panos),
        "n_sequences": int(panos["sequence_id"].nunique()),
        "n_creators": int(panos["creator_username"].nunique()),
        "capture_dates": [str(panos["capture_date"].min()), str(panos["capture_date"].max())],
        "camera_type": panos["camera_type"].value_counts(dropna=False).to_dict(),
        "all_true_equirect": bool((panos["width"] == 2 * panos["height"]).all()),
        "pano_heights": {int(h): int(n) for h, n in
                         panos["height"].value_counts().sort_index().items()},
        "rigs": rigs,
        "raw_field_degeneracy": {
            "compass_angle_exact_zero": int((panos["compass_angle"] == 0).sum()),
            "camera_parameters_present": int(panos["camera_parameters"].notna().sum()),
        },
        "sfm_vs_raw": {
            "position_shift_m": {"median": round(_pct(pos_shift, 50), 3),
                                 "p90": round(_pct(pos_shift, 90), 3),
                                 "max": round(float(np.max(pos_shift)), 1)},
            "abs_compass_delta_deg": {"median": round(_pct(compass_delta, 50), 2),
                                      "p90": round(_pct(compass_delta, 90), 2)},
            "altitude_delta_m": {"median": round(_pct(alt_delta, 50), 2),
                                 "p10": round(_pct(alt_delta, 10), 2),
                                 "p90": round(_pct(alt_delta, 90), 2)},
        },
        "quality_score": {"median": round(_pct(panos["quality_score"].dropna(), 50), 3),
                          "p10": round(_pct(panos["quality_score"].dropna(), 10), 3)},
        "capture_modes": mode_agg,
        "sequences_with_site_members": int((seq["n_site_members"] > 0).sum()),
    }


def census_gsv_run(run: str, data_dir: Path = DATA_DIR) -> dict:
    panos = load_panos(run, data_dir)
    members = site_member_counts(load_sites(run, data_dir))
    n_members_by_pano = panos["pano_id"].map(members).fillna(0).astype(int)
    by_height = {}
    for h, g in panos.assign(_members=n_members_by_pano).groupby("height"):
        by_height[int(h)] = {"n_panos": len(g), "n_site_members": int(g["_members"].sum())}
    return {
        "n_panos": len(panos),
        "capture_dates": [str(panos["capture_date"].min()), str(panos["capture_date"].max())],
        "pano_heights": by_height,
        "camera_pitch_deg": {"median": round(_pct(panos["camera_pitch"].dropna(), 50), 2),
                             "p90_abs": round(_pct(np.abs(panos["camera_pitch"].dropna()), 90), 2)},
        "camera_roll_deg": {"median": round(_pct(panos["camera_roll"].dropna(), 50), 2),
                            "p90_abs": round(_pct(np.abs(panos["camera_roll"].dropna()), 90), 2)},
    }


def build_census(data_dir: Path = DATA_DIR) -> dict:
    return {
        "mapillary": {run: census_mapillary_run(run, data_dir) for run in MAPILLARY_RUNS},
        "gsv_control": {run: census_gsv_run(run, data_dir) for run in GSV_RUNS},
    }


# --------------------------------------------------------------------------------------
# The two scale-free falsification diagnostics (#4766's method, reimplemented)
#
# Multi-view self-consistency: the same physical curb ramp seen from several panoramas
# should land in one place. Each candidate model places every member detection from its
# own panorama (SfM pose, member bearing, the model's predicted ground distance); the
# site consensus is the plain mean of those placements, so residuals are demeaned within
# site by construction and the whole scoring is anchored to a mean of the same rays —
# which is precisely why it can identify functional form but provably not absolute scale
# (RampNet#101). Everything reported is therefore scale-invariant:
#
# - rms_over_range   — 2D residual RMS divided by the mean predicted range
# - range_slope      — OLS of the along-ray residual on the predicted range, both
#                      within-site demeaned (m/m). Large negative = compression: far
#                      views under-shoot relative to their peers.
# - height_slope     — OLS of the along-ray residual (per site mean range) on
#                      pano_height/6656, both within-site demeaned. The "confound
#                      floor" is the height-normalized model's slope: it has no pixel
#                      dependence, so whatever slope it shows is rig confounding, not
#                      estimator error (#4765's diagnostic).
#
# The fuse gate (max range 25 m at 2.6 m camera height) means every member sits at
# depression >= ~5.9 deg — the diagnostics population is conditioned to the regime where
# flat-ground geometry is defensible, exactly as #4766 cautions.
# --------------------------------------------------------------------------------------

EARTH_R = 6371008.8
EST_INTERCEPT = 18.6051843       # production zoom-1 coefficients (PanoDataService.toLatLng)
EST_PANO_Y_SLOPE = 0.0138947
EST_CANVAS_Y_SLOPE = 0.0011023
CANVAS_CENTER_Y = 240.0          # canvas centre; reproduces #4765's worked table exactly
CALIBRATION_HEIGHT = 6656.0
COT_CAMERA_HEIGHT = 2.6          # ecosystem constant; reproduces member range_m exactly
BLEND_H_M = 2.783                # the shipped blend's CurbRamp height (#12 report §7)
BLEND_A_DEG = 11.25
BLEND_CAP_M = 50.0
MIN_SITE_MEMBERS = 2

MODEL_KEYS = ["A_status_quo", "B_normalized", "C_cotangent", "D_blend"]


def model_distances(dep_deg: np.ndarray, pano_height: np.ndarray) -> dict[str, np.ndarray]:
    """Predicted ground distance (m) for every candidate, from depression angle alone.

    dep_deg is the exact resolution-independent depression below the horizon
    (positive down); pano_height only feeds the pixel-frame models A and B.
    """
    dep = np.asarray(dep_deg, dtype=float)
    h = np.asarray(pano_height, dtype=float)
    offset_px = dep / 180.0 * h          # pixels below the horizon in the pano's own frame
    canvas_term = EST_CANVAS_Y_SLOPE * CANVAS_CENTER_Y
    a_rad = np.radians(BLEND_A_DEG)
    cot_at_a = BLEND_H_M / np.tan(a_rad)
    tail_slope = BLEND_H_M * (np.pi / 180.0) / np.sin(a_rad) ** 2
    with np.errstate(divide="ignore"):
        cot = np.where(dep > 0, COT_CAMERA_HEIGHT / np.tan(np.radians(dep)), np.inf)
    return {
        "A_status_quo": np.maximum(
            0.0, EST_INTERCEPT - EST_PANO_Y_SLOPE * offset_px + canvas_term),
        "B_normalized": np.maximum(
            0.0, EST_INTERCEPT - EST_PANO_Y_SLOPE * dep / 180.0 * CALIBRATION_HEIGHT
            + canvas_term),
        "C_cotangent": cot,
        # The tail evaluates at max(dep, 0) — the #12 review's horizon clamp, so a click
        # above the horizon gets the horizon's ~28 m, not a runaway extrapolation. Inert
        # here (the fuse gate keeps every member at dep >= ~5.9 deg) but kept identical
        # to the shipped form.
        "D_blend": np.where(
            dep >= BLEND_A_DEG,
            BLEND_H_M / np.tan(np.radians(np.maximum(dep, 1e-9))),
            np.clip(cot_at_a + tail_slope * (BLEND_A_DEG - np.maximum(dep, 0.0)),
                    0.0, BLEND_CAP_M)),
    }


def member_frame(run: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per in_refit member of every multi-member site, with pano metadata joined."""
    panos = load_panos(run, data_dir).set_index("pano_id")
    rows = []
    for site in load_sites(run, data_dir):
        members = [m for m in site["members"] if m.get("in_refit", True)]
        if len(members) < MIN_SITE_MEMBERS:
            continue
        for m in members:
            p = panos.loc[m["pano_id"]]
            rows.append({
                "site_id": site["site_id"],
                "pano_id": m["pano_id"],
                "sequence_id": p.get("sequence_id"),
                "rig": f"{p.get('camera_make') or '(none)'} / {p.get('camera_model') or '(none)'}",
                "pano_height": int(p["height"]),
                "pano_lat": float(p["lat"]),
                "pano_lng": float(p["lng"]),
                "y_normalized": float(m["y_normalized"]),
                "bearing_deg": float(m["bearing_deg"]),
                "range_m": float(m["range_m"]),
                "member_lat": float(m["lat"]),
                "member_lng": float(m["lng"]),
            })
    frame = pd.DataFrame(rows)
    frame["dep_deg"] = (frame["y_normalized"] - 0.5) * 180.0
    return frame


def _local_en(lat, lng, lat0: float, lng0: float) -> tuple[np.ndarray, np.ndarray]:
    east = np.radians(np.asarray(lng) - lng0) * np.cos(np.radians(lat0)) * EARTH_R
    north = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return east, north


def _demeaned_ols(y: np.ndarray, x: np.ndarray, groups: np.ndarray) -> dict:
    """Slope +/- naive OLS SE of y on x, both demeaned within group (no intercept left)."""
    d = pd.DataFrame({"y": y, "x": x, "g": groups})
    d["y"] -= d.groupby("g")["y"].transform("mean")
    d["x"] -= d.groupby("g")["x"].transform("mean")
    sxx = float((d["x"] ** 2).sum())
    if sxx < 1e-8 * len(d):
        # no real within-site variation in x (e.g. a run with a single pano height):
        # the slope is float noise over float noise, not a measurement
        return {"slope": None, "se": None}
    slope = float((d["x"] * d["y"]).sum() / sxx)
    resid = d["y"] - slope * d["x"]
    dof = max(len(d) - d["g"].nunique() - 1, 1)
    se = float(np.sqrt((resid ** 2).sum() / dof / sxx))
    return {"slope": round(slope, 4), "se": round(se, 4)}


def diagnose_run(run: str, data_dir: Path = DATA_DIR,
                 frame: pd.DataFrame | None = None,
                 scale: pd.Series | None = None) -> dict:
    """Score every candidate's self-consistency on one run's fused sites.

    scale: optional per-member distance multiplier (indexed like frame), used by the
    per-sequence camera-height stage to re-score D with fitted sequence scales.
    """
    f = member_frame(run, data_dir) if frame is None else frame
    lat0, lng0 = float(f["pano_lat"].mean()), float(f["pano_lng"].mean())
    pe, pn = _local_en(f["pano_lat"], f["pano_lng"], lat0, lng0)
    theta = np.radians(f["bearing_deg"])
    ux, uy = np.sin(theta), np.cos(theta)
    groups = f["site_id"].to_numpy()
    heights = f["pano_height"].to_numpy()

    out: dict = {"n_sites": int(f["site_id"].nunique()), "n_members": int(len(f)),
                 "per_model": {}}
    for key, dist in model_distances(f["dep_deg"].to_numpy(), heights).items():
        if scale is not None and key == "D_blend":
            dist = dist * scale.to_numpy()
        px, py = pe + dist * ux, pn + dist * uy
        d = pd.DataFrame({"px": px, "py": py, "g": groups, "dist": dist})
        cx = d.groupby("g")["px"].transform("mean")
        cy = d.groupby("g")["py"].transform("mean")
        ex, ey = d["px"] - cx, d["py"] - cy
        along = ex * ux + ey * uy
        # A can clamp to literal 0 m for a whole near-field site; floor the per-site
        # normalizer so those sites contribute zeros rather than infinities.
        mean_range = d.groupby("g")["dist"].transform("mean").clip(lower=0.5)
        out["per_model"][key] = {
            "rms_over_range": round(float(np.sqrt(np.mean(ex ** 2 + ey ** 2))
                                          / np.mean(dist)), 4),
            "range_slope": _demeaned_ols(along.to_numpy(), dist, groups),
            "height_slope": _demeaned_ols((along / mean_range).to_numpy(),
                                          heights / CALIBRATION_HEIGHT, groups),
        }
    return out


def conventions_check(run: str = "richmond", data_dir: Path = DATA_DIR) -> dict:
    """The auto-labeler's stored ray range must equal our cotangent at h=2.6 exactly."""
    f = member_frame(run, data_dir)
    cot = model_distances(f["dep_deg"].to_numpy(), f["pano_height"].to_numpy())["C_cotangent"]
    return {"max_abs_range_m_delta": round(float(np.max(np.abs(cot - f["range_m"]))), 6),
            "n": int(len(f))}


def build_diagnostics(data_dir: Path = DATA_DIR) -> dict:
    return {
        "conventions": conventions_check("richmond", data_dir),
        "runs": {run: diagnose_run(run, data_dir) for run in MAPILLARY_RUNS + GSV_RUNS},
    }


# --------------------------------------------------------------------------------------
# Per-sequence camera heights (the scale axis)
#
# The blend's h[t] was fit on GSV's ~2.6 m car. A Mapillary sequence is one rig on one
# outing, so a per-sequence multiplicative scale k on the predicted distance is exactly a
# per-sequence camera height h_seq = k * BLEND_H_M. We solve all k jointly by alternating
# least squares on the multi-view objective (place members, take site means, refit each
# sequence's k in closed form against the consensus of everyone else's placements).
#
# What is and is not identified: sites seen by two or more sequences pin the *relative*
# scale between those sequences; the global scale rides the same self-consistency trap as
# RampNet#101 (shrinking every range buys residual for free when views cluster on one
# side), so everything reported is relative to the run's member-weighted geometric mean,
# and the absolute anchor stays the GSV-fit heights.
# --------------------------------------------------------------------------------------

SEQ_SCALE_ITERATIONS = 200
SEQ_MIN_MEMBERS = 5


def fit_sequence_scales(run: str, data_dir: Path = DATA_DIR,
                        frame: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Alternating least squares for per-sequence distance scales under D_blend.

    Returns (member frame with fitted per-member scale, per-sequence table).
    """
    f = member_frame(run, data_dir) if frame is None else frame
    lat0, lng0 = float(f["pano_lat"].mean()), float(f["pano_lng"].mean())
    pe, pn = _local_en(f["pano_lat"], f["pano_lng"], lat0, lng0)
    theta = np.radians(f["bearing_deg"])
    ux, uy = np.sin(theta), np.cos(theta)
    d0 = model_distances(f["dep_deg"].to_numpy(), f["pano_height"].to_numpy())["D_blend"]
    seq_codes, seq_ids = pd.factorize(f["sequence_id"])
    site_codes, _ = pd.factorize(f["site_id"])
    k = np.ones(len(seq_ids))
    n_sites = site_codes.max() + 1
    for _ in range(SEQ_SCALE_ITERATIONS):
        dist = k[seq_codes] * d0
        px, py = pe + dist * ux, pn + dist * uy
        counts = np.bincount(site_codes, minlength=n_sites)
        cx = np.bincount(site_codes, weights=px, minlength=n_sites) / counts
        cy = np.bincount(site_codes, weights=py, minlength=n_sites) / counts
        tx, ty = cx[site_codes] - pe, cy[site_codes] - pn
        num = np.bincount(seq_codes, weights=d0 * (tx * ux + ty * uy),
                          minlength=len(seq_ids))
        den = np.bincount(seq_codes, weights=d0 ** 2, minlength=len(seq_ids))
        k_new = np.where(den > 0, num / np.maximum(den, 1e-12), 1.0)
        if np.max(np.abs(k_new - k)) < 1e-10:
            k = k_new
            break
        k = k_new

    # relative to the member-weighted geometric mean (the unidentified global axis)
    member_k = k[seq_codes]
    k_rel = k / np.exp(np.mean(np.log(np.maximum(member_k, 1e-6))))
    seq_table = pd.DataFrame({
        "sequence_id": seq_ids,
        "k_rel": k_rel[np.arange(len(seq_ids))],
        "n_members": np.bincount(seq_codes, minlength=len(seq_ids)),
    })
    rig_by_seq = f.groupby("sequence_id")["rig"].first()
    seq_table["rig"] = seq_table["sequence_id"].map(rig_by_seq)
    seq_table["implied_h_m"] = seq_table["k_rel"] * BLEND_H_M
    f = f.assign(seq_scale=member_k / np.exp(np.mean(np.log(np.maximum(member_k, 1e-6)))))
    return f, seq_table


def sequence_scale_summary(run: str, data_dir: Path = DATA_DIR) -> dict:
    f, seq_table = fit_sequence_scales(run, data_dir)

    # identifiability: how many sites see more than one sequence?
    per_site = f.groupby("site_id")["sequence_id"].nunique()
    fitted = seq_table[seq_table["n_members"] >= SEQ_MIN_MEMBERS]

    per_rig = {}
    for rig, g in fitted.groupby("rig"):
        weights = g["n_members"].to_numpy(dtype=float)
        per_rig[rig] = {
            "n_sequences": int(len(g)),
            "n_members": int(g["n_members"].sum()),
            "k_rel_median": round(float(g["k_rel"].median()), 4),
            "k_rel_iqr": [round(float(g["k_rel"].quantile(0.25)), 4),
                          round(float(g["k_rel"].quantile(0.75)), 4)],
            "k_rel_weighted_mean": round(float(np.average(g["k_rel"], weights=weights)), 4),
        }

    # does the fitted scale actually buy self-consistency, and does it fix the range axis?
    base = diagnose_run(run, data_dir, frame=f)["per_model"]["D_blend"]
    scaled = diagnose_run(run, data_dir, frame=f,
                          scale=f["seq_scale"])["per_model"]["D_blend"]
    return {
        "n_sequences_fitted": int(len(fitted)),
        "n_multi_sequence_sites": int((per_site >= 2).sum()),
        "n_sites": int(per_site.size),
        "per_rig": per_rig,
        "d_blend_unscaled": base,
        "d_blend_per_sequence_scale": scaled,
    }


def build_sequence_scales(data_dir: Path = DATA_DIR) -> dict:
    return {run: sequence_scale_summary(run, data_dir) for run in MAPILLARY_RUNS}
