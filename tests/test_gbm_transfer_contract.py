"""Contract tests for the #6 transfer harness: the frame mapping, the split, the recalibrations.

The findings tests assert what the run measured. These assert that the machinery which
produced it is the machinery it claims to be — and they run without fitting a booster, so
they stay cheap enough to be the first thing that breaks when something drifts.

The load-bearing one is the frame mapping. The era client stored ``sv_image_y`` as a
fixed-frame offset from the horizon; the modern schema stores ``pano_y``, an absolute row
in the panorama's real raster. Feeding the second into a booster trained on the first
would produce a plausible, entirely wrong transfer result — a silent failure with no
exception anywhere — so it is checked three ways: algebraically, against the era rows'
own real-pixel column, and against the discriminating alternative (the unmapped offset,
which must FAIL on 8192-px panoramas and pass on 6656-px ones).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import distance_refit as dr  # noqa: E402
import gbm_transfer as gt  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
PX_PER_DEG = gt.CALIBRATION_HEIGHT / 180.0


def _synthetic(dep_deg, pano_height):
    """A frame carrying only what to_era_frame needs, with pano_y placed at a known angle."""
    dep = np.asarray(dep_deg, float)
    h = np.asarray(pano_height, float)
    return pd.DataFrame({
        "pano_y": h / 2.0 + dep * h / 180.0,
        "pano_height": h,
        "canvas_x": np.full(len(dep), 360.0), "canvas_y": np.full(len(dep), 240.0),
        "heading": np.full(len(dep), 90.0), "pitch": np.full(len(dep), -10.0),
        "zoom": np.full(len(dep), 1.0),
    })


# ------------------------------------------------------------------------- frame mapping

def test_mapping_is_the_fixed_frame_offset():
    """sv_image_y = -depression * 6656/180, exactly, at every angle."""
    dep = np.array([-5.0, 0.0, 1.0, 5.0, 15.0, 45.0, 89.0])
    out = gt.to_era_frame(_synthetic(dep, np.full(len(dep), 8192.0)))
    assert np.allclose(out["sv_image_y"], -dep * PX_PER_DEG, atol=1e-9)


def test_mapping_is_resolution_invariant():
    """The whole point: the same click angle maps to the same column whatever the raster.

    An unmapped ``pano_height/2 - pano_y`` would differ by 8192/6656 = 1.23x between these
    two, which is the #4765 defect and would be a 23% distance error if it leaked in here.
    """
    dep = np.array([2.0, 8.0, 15.0, 30.0])
    a = gt.to_era_frame(_synthetic(dep, np.full(4, 6656.0)))["sv_image_y"].to_numpy()
    b = gt.to_era_frame(_synthetic(dep, np.full(4, 8192.0)))["sv_image_y"].to_numpy()
    assert np.allclose(a, b, atol=1e-9)
    raw = 8192 / 2 - _synthetic(dep, np.full(4, 8192.0))["pano_y"].to_numpy()
    assert not np.allclose(a, raw, atol=1.0)  # the discriminating alternative must differ


def _era_pixel_residuals(df: pd.DataFrame) -> dict:
    """Median (stored sv_image_y − candidate offset) per height group, for both candidates.

    ``mapped`` is the conversion this module ships; ``raw`` is the discriminating
    alternative (the unmapped ``pano_height/2 − pano_y``), which must fail at 8192 px.
    """
    out = {}
    for h, g in df[df["pano_height"].isin([6656, 8192])].groupby("pano_height"):
        pano_y = g["current_pano_y"].astype(float)
        raw = float(h) / 2.0 - pano_y
        mapped = raw * (gt.CALIBRATION_HEIGHT / float(h))
        out[int(h)] = {"n": int(len(g)),
                       "mapped": float((g["sv_image_y"] - mapped).median()),
                       "raw": float((g["sv_image_y"] - raw).median())}
    return out


def test_mapping_agrees_with_the_era_columns_own_real_pixels():
    """On real era rows, the mapped offset lands on the stored sv_image_y in BOTH height
    groups, and the unmapped one does so only at 6656 px.

    Two height groups agreeing to the same small residual is what makes this the right
    conversion rather than a fitted fudge: the residual is pano re-registration drift
    between the era panorama and today's, and it is the same ~15 px either way. Read
    straight from one committed city so the test costs a second, not a pipeline run.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "labels-seattle-latlng.csv.gz"),
                     usecols=["sv_image_y", "current_pano_y", "pano_height"],
                     low_memory=False).dropna()
    residuals = _era_pixel_residuals(df)
    assert set(residuals) == {6656, 8192}
    for h, r in residuals.items():
        assert abs(r["mapped"]) < 40.0, (h, r)          # ~1 degree of pano drift, no more
    assert abs(residuals[6656]["mapped"] - residuals[8192]["mapped"]) < 10.0
    assert abs(residuals[8192]["raw"]) > 100.0          # the wrong frame is not subtle


# The report's §3 table publishes this check's numbers over the whole era analysis frame,
# so they are locked here rather than left as prose. Recomputed by the RUN_SLOW test below;
# the fast Seattle test above is the cheap early-warning version of the same comparison.
ERA_FRAME_RESIDUALS_PX = {
    "n_rows": 162846,
    6656: {"n": 37494, "mapped": 15.000, "raw": 15.000},
    8192: {"n": 125352, "mapped": 14.562, "raw": 140.000},
}


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="loads and cleans the whole era frame (~1 min, several GB); set RUN_SLOW=1",
)
def test_era_frame_residuals_match_the_published_table():
    """The report's third frame-mapping check, recomputed from the committed CSVs.

    §3 of `reports/2026-08-10-gbm-transfer.md` publishes exact figures for this — 162,846
    era rows, +15.0 px at 6656 and +14.6 px at 8192, against +140 px for the unmapped
    offset at 8192 — and a published number with no artifact behind it is the defect this
    repo's archival rule exists to prevent. The population is the *cleaned* analysis frame
    (what the boosters are fitted on), restricted to rows carrying a current real-pixel
    row, which is what makes it 162,846 rather than the 195,242 raw rows on disk.
    """
    from label_latlng_estimation import clean_data, load_data  # noqa: PLC0415

    cleaned, _ = clean_data(load_data(DATA_DIR))
    df = cleaned.dropna(subset=["sv_image_y", "current_pano_y", "pano_height"])
    residuals = _era_pixel_residuals(df)

    assert set(residuals) == {6656, 8192}
    assert (sum(r["n"] for r in residuals.values())
            == ERA_FRAME_RESIDUALS_PX["n_rows"])
    for h in (6656, 8192):
        want = ERA_FRAME_RESIDUALS_PX[h]
        assert residuals[h]["n"] == want["n"], (h, residuals[h])
        assert abs(residuals[h]["mapped"] - want["mapped"]) < 5e-3, (h, residuals[h])
        assert abs(residuals[h]["raw"] - want["raw"]) < 5e-3, (h, residuals[h])
    # the whole point: one small drift in both groups for the mapping, a 23% frame error
    # for the alternative, and only at the resolution that discriminates them
    assert abs(residuals[6656]["mapped"] - residuals[8192]["mapped"]) < 1.0
    assert residuals[8192]["raw"] > 100.0


def test_frame_mapping_evidence_reports_both_checks():
    dep = np.array([3.0, 9.0, 20.0])
    out = gt.to_era_frame(_synthetic(dep, np.full(3, 8192.0)))
    out["depression_deg"] = dep
    ev = gt.frame_mapping_evidence(out)
    assert ev["n"] == 3
    assert ev["max_abs_diff_vs_pixel_angle_px"] < 1e-6
    assert abs(ev["px_per_deg"] - PX_PER_DEG) < 1e-12


def test_the_boosters_cannot_see_the_truth():
    """A leakage guard with teeth: no truth-bearing or estimator-output column may appear
    in the feature list the boosters are built from."""
    forbidden = {"truth_m", "truth_range_m", "stored_dist_m", "flat_earth_m", "pano_dist",
                 "A_deployed", "B_normalized", "C_anchor", "D_blend", "lat", "lng"}
    import run_gbm_ceiling as gc
    assert not (set(gc.FEATURES_FULL) | {"depression_deg"}) & forbidden


# ----------------------------------------------------------------------------- the split

def test_pano_half_split_reproduces_the_committed_stage4_split():
    """Same seed, same rows as modern_truth.remedy_check — the reason this module's
    closed-form rows can be asserted equal to the committed Stage 4 remedy table."""
    import json

    human = gt.gated_human(gt.load_modern_labels(DATA_DIR))
    in_train, in_test = gt.pano_half_split(human)
    with open(os.path.join(DATA_DIR, "modern-truth-summary.json"), encoding="utf-8") as f:
        split = json.load(f)["remedies"]["split"]
    assert int(in_train.sum()) == split["n_train_rows"]
    assert int(in_test.sum()) == split["n_test_rows"]
    assert not (in_train & in_test).any() and (in_train | in_test).all()
    # split BY panorama: no panorama may straddle the halves
    assert set(human["pano_id"][in_train]).isdisjoint(set(human["pano_id"][in_test]))


# -------------------------------------------------------------------------- recalibration

def test_scale_factor_recovers_a_planted_scale():
    rng = np.random.default_rng(0)
    truth = rng.uniform(3.0, 30.0, 4000)
    dep = np.full(4000, 20.0)
    assert abs(gt.scale_factor(truth / 1.25, truth, dep) - 1.25) < 1e-9


def test_scale_factor_respects_the_depression_floor():
    """Rows below the floor must not vote — they are where tan(dep) makes the ratio wild."""
    truth = np.concatenate([np.full(500, 10.0), np.full(500, 99.0)])
    pred = np.concatenate([np.full(500, 5.0), np.full(500, 1.0)])
    dep = np.concatenate([np.full(500, 20.0), np.full(500, 1.0)])
    assert abs(gt.scale_factor(pred, truth, dep) - 2.0) < 1e-9


def test_affine_l1_recovers_planted_coefficients():
    x = np.linspace(1.0, 40.0, 500)
    a, b = gt.affine_l1(x, -0.8 + 1.15 * x)
    assert abs(a + 0.8) < 1e-3 and abs(b - 1.15) < 1e-4


def test_quantile_map_is_monotone_and_matches_the_train_marginal():
    rng = np.random.default_rng(1)
    pred = rng.gamma(4.0, 2.0, 3000)
    truth = 1.3 * pred + 0.5
    f = gt.quantile_map(pred, truth)
    probe = np.sort(rng.gamma(4.0, 2.0, 400))
    assert np.all(np.diff(f(probe)) >= -1e-9)                       # monotone
    assert abs(np.median(f(pred)) - np.median(truth)) < 1e-6        # marginals match


def test_implied_height_recovers_a_planted_camera():
    dep = np.tile([8.0, 15.0, 30.0], 200)
    df = pd.DataFrame({"depression_deg": dep,
                       "pano_height": np.tile([6656.0, 8192.0, 8192.0], 200),
                       "truth_m": 2.5 / np.tan(np.radians(dep))})
    out = gt.implied_height_by_resolution(df, "truth_m", "depression_deg")
    assert abs(out["pooled"]["implied_height_m"] - 2.5) < 1e-9
    for group in out["by_pano_height"].values():
        assert abs(group["implied_height_m"] - 2.5) < 1e-9


def test_implied_height_separates_two_planted_cameras():
    """The diagnostic must actually resolve a per-resolution scale split, not average it."""
    n = 600
    dep = np.tile([10.0, 20.0], n)
    h = np.repeat([6656.0, 8192.0], n)
    truth = np.where(h == 6656.0, 2.8, 2.35) / np.tan(np.radians(dep))
    out = gt.implied_height_by_resolution(
        pd.DataFrame({"depression_deg": dep, "pano_height": h, "truth_m": truth}),
        "truth_m", "depression_deg")
    assert abs(out["by_pano_height"]["6656"]["implied_height_m"] - 2.8) < 1e-9
    assert abs(out["by_pano_height"]["8192"]["implied_height_m"] - 2.35) < 1e-9


# ------------------------------------------------------------------------------ bootstrap

def test_bootstrap_clusters_by_panorama():
    """Resampling panoramas, not rows: a dataset that is one panorama repeated has no
    between-cluster variation, so its interval must collapse to a point."""
    n = 400
    preds = pd.DataFrame({"m": np.full(n, 5.0)})
    truth = np.full(n, 4.0)
    one = gt.bootstrap_medians(preds, truth, np.full(n, "p1"), ["m"], n_boot=50)
    assert one["n_panos"] == 1
    assert one["ci"]["m"]["median_abs_m_hi"] - one["ci"]["m"]["median_abs_m_lo"] < 1e-9

    many = gt.bootstrap_medians(preds, truth, np.arange(n).astype(str), ["m"], n_boot=50)
    assert many["n_panos"] == n


def test_bootstrap_paired_difference_tracks_the_point_estimate():
    rng = np.random.default_rng(2)
    n = 800
    truth = rng.uniform(4.0, 20.0, n)
    preds = pd.DataFrame({"good": truth + rng.normal(0, 0.2, n),
                          "bad": truth + rng.normal(0, 2.0, n)})
    out = gt.bootstrap_medians(preds, truth, np.repeat(np.arange(80), 10).astype(str),
                               ["good", "bad"], reference="good", n_boot=200)
    diff = out["paired_diff_vs_reference"]["bad"]
    assert diff["delta_median_abs_m_lo"] > 0            # 'bad' is worse, decisively
    assert diff["frac_draws_better_than_reference"] == 0.0


def test_error_vs_distance_drops_thin_bins():
    truth = np.concatenate([np.full(200, 7.0), np.full(3, 40.0)])
    preds = pd.DataFrame({"m": truth + 0.5})
    rows = gt.error_vs_distance(preds, truth, ["m"])
    assert [r["bin_m"] for r in rows] == ["[5.0, 10.0)"]
    assert abs(rows[0]["per_model"]["m"]["median_abs_m"] - 0.5) < 1e-9


def test_label_type_census_flags_the_types_the_booster_never_saw():
    df = pd.DataFrame({"label_type": ["CurbRamp"] * 8 + ["Crosswalk"] * 2})
    census = gt.label_type_census(df)
    assert census["unseen_types"] == ["Crosswalk"]
    assert census["n_unseen_rows"] == 2
    assert abs(census["frac_unseen_rows"] - 0.2) < 1e-12
    assert "CurbRamp" in census["era_categories"]
    assert "Crosswalk" not in dr.LABEL_TYPES  # the premise of the whole cut


@pytest.mark.parametrize("key", ["A_deployed", "D_blend"])
def test_closed_form_predictions_come_from_the_committed_columns(key):
    """A_deployed and D_blend are read, not recomputed: they must be the very numbers
    run_modern_truth scored, or the comparison is against a different model."""
    human = gt.gated_human(gt.load_modern_labels(DATA_DIR))
    preds = gt.closed_form_predictions(human, DATA_DIR)
    assert np.allclose(preds[key].to_numpy(float), human[key].to_numpy(float))
