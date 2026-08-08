"""Entry point for issue #7 — bearing-only triangulation.

Builds ``data/triangulation-summary.json`` from the committed auto-labeler multi-view
inputs (``data/falsification-*``). Offline: no network, no database.

    python python/run_triangulation.py build --write
    python python/run_triangulation.py build            # print, do not write

Stages, in the order the report reads them:

  ``applicability``  how many objects are multiply observed, and how well conditioned
  ``noise``          the error budget: sigma_bearing / sigma_pos per run, converged
  ``validation``     synthetic + real-geometry bias checks on the estimator itself
  ``scale``          the headline: implied camera height, per run, with no depth data
  ``robustness``     conditioning sweep, site-size sweep, fuse-gate selection probe
  ``scoring``        every distance model against the triangulated truth
  ``cross_source``   rig heights across imagery sources, where the click convention cancels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import triangulation as tg  # noqa: E402
import triangulation_depth as td  # noqa: E402

OUT = tg.DATA_DIR / "triangulation-summary.json"

#: The modern GSV rig height measured from depth on human clicks (modern-truth close-out).
#: Used only as a *comparand* — nothing here is fit to it.
SHIPPED_HEIGHT_M = 2.341219672825709


def build(data_dir: Path = tg.DATA_DIR, runs=None, quick: bool = False) -> dict:
    runs = list(runs or tg.ALL_RUNS)
    out: dict = {
        "meta": {
            "issue": "https://github.com/ProjectSidewalk/label-latlng-estimation/issues/7",
            "inputs": "data/falsification-sites-*.jsonl.gz + data/falsification-panos-*.csv.gz",
            "estimator": "leave-one-out least-squares intersection of bearing rays",
            "uses": "panorama positions, panorama headings, horizontal detection angle",
            "does_not_use": ("vertical click angle, camera height, ground-plane assumption, "
                             "depth data, panorama resolution"),
            "shipped_height_m": SHIPPED_HEIGHT_M,
            "assumed_height_m": tg.COT_CAMERA_HEIGHT,
            "sigma_gate_m": tg.SIGMA_R_GATE_M,
            "min_panos_for_loo": tg.MIN_PANOS_FOR_LOO,
            "seed": tg.SEED,
        },
        "imagery": {r: ("mapillary" if r in tg.MAPILLARY_RUNS else "gsv") for r in runs},
        "applicability": {}, "noise": {}, "scale": {}, "scale_global": {},
        "robustness": {},
        "scoring": {}, "by_range": {}, "by_depression": {}, "split_half": {},
        "bearing_offset": {},
    }

    # --- estimator validation, independent of any run -------------------------------
    out["validation"] = {
        "synthetic": {
            f"sigma_bearing_{sb}_sigma_pos_{sp}": tg.monte_carlo_bias_check(
                sigma_pos_m=sp, sigma_bearing_deg=sb,
                n_trials=4000 if quick else 20000)
            for sb, sp in [(0.5, 0.5), (1.4, 0.5), (1.4, 1.0)]
        },
    }

    frames = {}
    for run in runs:
        print(f"  [{run}] fitting noise ...", flush=True)
        fit = tg.fit_noise(run, data_dir)
        f = fit.pop("frame")
        frames[run] = f
        out["noise"][run] = fit
        out["applicability"][run] = tg.applicability(run, data_dir)
        out["scale"][run] = tg.implied_height(f, n_boot=120 if quick else 400)
        # The preferred estimator: one global scale fitted on multi-view agreement,
        # never dividing by a noisy triangulated range.
        out["scale_global"][run] = tg.fit_model_scale(f, n_boot=0 if quick else 200)
        out["robustness"][run] = {
            "sensitivity": tg.scale_sensitivity(f),
            "fuse_gate_selection": tg.selection_probe(f),
            "camera_tilt_hypothesis": tg.tilt_probe(run, f, data_dir),
            "by_range_m": tg.implied_height_by(f, "r_tri", bins=(0, 5, 10, 15, 20, 60)),
            "by_depression_deg": tg.implied_height_by(
                f, "dep_deg", bins=(0, 6, 9, 12, 16, 22, 90)),
        }
        out["scoring"][run] = tg.score_models(f, data_dir)
        out["by_range"][run] = tg.score_by_range(f, data_dir)
        out["split_half"][run] = tg.split_half_precision(f)
        if not quick:
            print(f"  [{run}] bearing offset ...", flush=True)
            out["bearing_offset"][run] = tg.fit_bearing_offset(run, data_dir)
            print(f"  [{run}] parametric bootstrap ...", flush=True)
            out["validation"].setdefault("real_geometry", {})[run] = \
                tg.parametric_bootstrap_bias(
                    f, fit["sigma_bearing_deg"], fit["sigma_pos_m"],
                    height_m=SHIPPED_HEIGHT_M)

    out["cross_source"] = cross_source(out, runs)
    # The depth anchor replays committed payloads; absent them it records why it is absent
    # rather than silently omitting the section.
    print("  depth anchor ...", flush=True)
    out["depth_anchor"] = td.anchor(data_dir)
    return out


def cross_source(out: dict, runs) -> dict:
    """Rig heights across imagery sources, with the detector's click convention cancelled.

    ``implied_height`` measures ``H_rig - delta``, where ``delta`` is however far above the
    ground contact the detector's own click point sits. ``delta`` is a property of the
    detector, and the auto-labeler ran *the same* detector on every run — so it cancels in
    a *difference* between runs even though it is not identifiable within one.

    That makes the GSV runs a ruler: their measured ``H - delta``, set against the
    independently measured modern GSV rig height, calibrates ``delta`` once, and every
    Mapillary run's absolute rig height then follows. Nothing else in this repository can
    produce an absolute Mapillary camera height — Stage 3's per-sequence heights were
    relative by construction.
    """
    gsv = [r for r in runs if r in tg.GSV_RUNS
           and out["scale_global"].get(r, {}).get("height_m")]
    if not gsv:
        return {}
    gsv_vals = [out["scale_global"][r]["height_m"] for r in gsv]
    gsv_pooled = float(np.median(gsv_vals))
    delta = SHIPPED_HEIGHT_M - gsv_pooled     # + means the click sits above ground contact
    res = {
        "gsv_runs": gsv,
        "gsv_implied_heights_m": {r: out["scale_global"][r]["height_m"] for r in gsv},
        "gsv_pooled_implied_m": round(gsv_pooled, 4),
        "shipped_gsv_rig_m": SHIPPED_HEIGHT_M,
        "detector_click_offset_m": round(delta, 4),
        "detector_click_offset_note": (
            "H_rig - implied. Positive = the detector's click point sits above the ground "
            "contact. Calibrated on the GSV runs against the depth-measured modern rig, "
            "then transferred; it is NOT identifiable from bearings alone."),
        "absolute_rig_heights_m": {},
    }
    for r in runs:
        s = out["scale_global"].get(r, {})
        if s.get("height_m") is None:
            continue
        res["absolute_rig_heights_m"][r] = {
            "implied_m": s["height_m"],
            "absolute_rig_m": round(s["height_m"] + delta, 4),
            "imagery": out["imagery"][r],
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["build", "fetch"], nargs="?", default="build")
    ap.add_argument("--write", action="store_true", help="write data/triangulation-summary.json")
    ap.add_argument("--quick", action="store_true", help="skip the slow validation stages")
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--per-run", type=int, default=td.PANOS_PER_RUN,
                    help="fetch: panoramas sampled per GSV run")
    args = ap.parse_args()

    if args.stage == "fetch":
        # The only stage that touches the network. Its payloads are committed verbatim,
        # so every later stage replays from a fresh checkout.
        print(json.dumps(td.fetch(runs=args.runs, per_run=args.per_run), indent=2))
        return 0

    summary = build(runs=args.runs, quick=args.quick)
    if args.write:
        OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8", newline="\n")
        print(f"wrote {OUT}")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
