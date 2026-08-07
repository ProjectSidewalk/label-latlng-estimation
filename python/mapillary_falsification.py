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
