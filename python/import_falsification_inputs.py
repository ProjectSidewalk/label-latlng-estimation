"""Import the Stage 3 (Mapillary falsification) inputs from a sidewalk-auto-labeler checkout.

Issue #3 Stage 3 falsifies the refit candidates on imagery with no depth ground truth, using
multi-view self-consistency over the auto-labeler's fused curb-ramp sites. Those inputs —
``runs/<run>/sites.jsonl`` (fuse_sites.py output) and the per-pano metadata inside
``runs/<run>/results.jsonl`` — are gitignored in the auto-labeler repo, so this script preserves
them here, in ``data/``, the same way the depth payloads were preserved for #4/#9: commit the
bytes, keep the analysis reproducible offline forever.

For each run it writes:

- ``data/falsification-sites-<run>.jsonl.gz`` — the sites.jsonl verbatim (deterministic gzip).
- ``data/falsification-panos-<run>.csv.gz`` — one row per panorama from results.jsonl: position,
  dimensions, pose, and (Mapillary) the full Graph API source_metadata census fields
  (camera_type/make/model, sequence, raw vs computed geometry/compass/altitude, SfM rotation,
  creator, quality_score). GSV rows leave the Mapillary-only columns empty.

and one combined ``data/falsification-runs-meta.json``: per-run fuse parameters and counts
(from sites_meta.json), source-file SHA-256s, per-run height×width histograms, and the
auto-labeler commit the runs directory was at.

Usage:
    python python/import_falsification_inputs.py D:/Git/sidewalk-auto-labeler/runs

Deterministic: given identical inputs the outputs are byte-identical (gzip mtime=0, sorted JSON
keys). Network is never touched.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import subprocess
from collections import Counter
from pathlib import Path

MAPILLARY_RUNS = ["richmond", "clovis"]
GSV_RUNS = ["paterson", "gainesville", "bend", "sao_paulo"]
RUNS = MAPILLARY_RUNS + GSV_RUNS

PANO_COLUMNS = [
    "run", "source", "pano_id", "sequence_id", "capture_date", "captured_at_ms",
    "width", "height", "lat", "lng", "camera_heading", "camera_pitch", "camera_roll",
    "n_detections",
    "camera_make", "camera_model", "camera_type", "camera_parameters", "quality_score",
    "altitude", "computed_altitude", "atomic_scale",
    "compass_angle", "computed_compass_angle", "raw_lat", "raw_lng",
    "computed_rotation_x", "computed_rotation_y", "computed_rotation_z",
    "creator_username", "creator_id", "exif_orientation", "merge_cc", "copyright",
]


def _fmt(value):
    """CSV cell for a JSON value: shortest round-trip for floats, empty for None."""
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gzip_bytes(payload: bytes) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        gz.write(payload)
    return buf.getvalue()


def pano_row(run: str, record: dict) -> dict:
    pano = record["pano"]
    sm = pano.get("source_metadata") or {}
    raw_geom = (sm.get("geometry") or {}).get("coordinates") or [None, None]
    rotation = sm.get("computed_rotation") or [None, None, None]
    creator = sm.get("creator") or {}
    return {
        "run": run,
        "source": pano.get("source"),
        "pano_id": pano["panorama_id"],
        "sequence_id": pano.get("sequence_id"),
        "capture_date": pano.get("capture_date"),
        "captured_at_ms": sm.get("captured_at"),
        "width": pano.get("width"),
        "height": pano.get("height"),
        "lat": pano.get("lat"),
        "lng": pano.get("lng"),
        "camera_heading": pano.get("camera_heading"),
        "camera_pitch": pano.get("camera_pitch"),
        "camera_roll": pano.get("camera_roll"),
        "n_detections": len(record.get("detections") or []),
        "camera_make": pano.get("camera_make"),
        "camera_model": pano.get("camera_model"),
        "camera_type": pano.get("camera_type"),
        "camera_parameters": sm.get("camera_parameters"),
        "quality_score": pano.get("quality_score"),
        "altitude": sm.get("altitude"),
        "computed_altitude": sm.get("computed_altitude"),
        "atomic_scale": sm.get("atomic_scale"),
        "compass_angle": sm.get("compass_angle"),
        "computed_compass_angle": sm.get("computed_compass_angle"),
        "raw_lat": raw_geom[1],
        "raw_lng": raw_geom[0],
        "computed_rotation_x": rotation[0],
        "computed_rotation_y": rotation[1],
        "computed_rotation_z": rotation[2],
        "creator_username": creator.get("username"),
        "creator_id": creator.get("id"),
        "exif_orientation": sm.get("exif_orientation"),
        "merge_cc": sm.get("merge_cc"),
        "copyright": pano.get("copyright"),
    }


def import_run(run: str, runs_root: Path, data_dir: Path) -> dict:
    run_dir = runs_root / run
    results_path = run_dir / "results.jsonl"
    sites_path = run_dir / "sites.jsonl"
    sites_meta_path = run_dir / "sites_meta.json"

    # sites.jsonl verbatim
    sites_bytes = sites_path.read_bytes()
    (data_dir / f"falsification-sites-{run}.jsonl.gz").write_bytes(_gzip_bytes(sites_bytes))

    # per-pano table from results.jsonl
    rows, dims, sources = [], Counter(), Counter()
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            row = pano_row(run, record)
            rows.append(row)
            dims[f"{row['width']}x{row['height']}"] += 1
            sources[row["source"]] += 1
    pano_ids = {r["pano_id"] for r in rows}
    if len(pano_ids) != len(rows):
        raise SystemExit(f"{run}: duplicate pano ids in results.jsonl")

    # every fused member must resolve in the pano table
    n_sites = n_members = 0
    for line in sites_bytes.decode("utf-8").splitlines():
        site = json.loads(line)
        n_sites += 1
        for member in site["members"]:
            n_members += 1
            if member["pano_id"] not in pano_ids:
                raise SystemExit(f"{run}: site member pano {member['pano_id']} missing from results.jsonl")

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(PANO_COLUMNS)
    for row in rows:
        writer.writerow([_fmt(row[c]) for c in PANO_COLUMNS])
    (data_dir / f"falsification-panos-{run}.csv.gz").write_bytes(
        _gzip_bytes(buf.getvalue().encode("utf-8")))

    print(f"{run}: {len(rows)} panos, {n_sites} sites / {n_members} members, "
          f"dims {dict(sorted(dims.items()))}")
    return {
        "imagery": "mapillary" if run in MAPILLARY_RUNS else "gsv",
        "n_panos": len(rows),
        "n_sites": n_sites,
        "n_site_members": n_members,
        "pano_dims": dict(sorted(dims.items())),
        "sources": dict(sorted(sources.items())),
        "fuse": json.loads(sites_meta_path.read_text(encoding="utf-8")),
        "results_sha256": _sha256(results_path),
        "sites_sha256": _sha256(sites_path),
    }


def auto_labeler_commit(runs_root: Path):
    try:
        out = subprocess.run(["git", "-C", str(runs_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_root", type=Path,
                        help="the auto-labeler checkout's runs/ directory")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()

    meta = {
        "_provenance": {
            "imported_from": "sidewalk-auto-labeler runs/ (sites.jsonl + results.jsonl per run)",
            "auto_labeler_commit": auto_labeler_commit(args.runs_root),
            "importer": "python/import_falsification_inputs.py",
            "issue": "https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3 (Stage 3)",
        },
        "runs": {run: import_run(run, args.runs_root, args.data_dir) for run in RUNS},
    }
    meta_path = args.data_dir / "falsification-runs-meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
