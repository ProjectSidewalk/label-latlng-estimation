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


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="loads and cleans the whole era frame (~1 min, several GB); set RUN_SLOW=1",
)
def test_era_frame_residuals_reproduce_the_committed_block():
    """The third frame-mapping check, recomputed from the committed CSVs.

    §3 of `reports/2026-08-10-gbm-transfer.md` publishes exact figures for this — 162,846
    era rows, +15.0 px at 6656 and +14.6 px at 8192, against +140 px for the unmapped
    offset at 8192. Those now live in the summary's `frame_mapping.era_frame_residuals_px`
    (the findings tests lock their values), so this test re-derives them from the CSVs and
    compares against the artifact rather than against a second hand-maintained copy — the
    artifact is the single source of truth for a published number.

    The population is the *cleaned* analysis frame (what the boosters are fitted on),
    restricted to rows carrying a current real-pixel row, which is what makes it 162,846
    rather than the 195,242 raw rows on disk. Recomputed here through this test's own
    `_era_pixel_residuals`, not `gt.era_pixel_residuals`, so the check stays independent
    of the function that wrote the block.
    """
    import json  # noqa: PLC0415

    from label_latlng_estimation import clean_data, load_data  # noqa: PLC0415

    with open(os.path.join(DATA_DIR, "gbm-transfer-summary.json"), encoding="utf-8") as f:
        committed = json.load(f)["frame_mapping"]["era_frame_residuals_px"]

    cleaned, _ = clean_data(load_data(DATA_DIR))
    df = cleaned.dropna(subset=["sv_image_y", "current_pano_y", "pano_height"])
    residuals = _era_pixel_residuals(df)

    assert set(residuals) == {6656, 8192}
    assert sum(r["n"] for r in residuals.values()) == committed["n_rows"]
    for h in (6656, 8192):
        want = committed["by_pano_height"][str(h)]
        assert residuals[h]["n"] == want["n"], (h, residuals[h])
        assert abs(residuals[h]["mapped"] - want["mapped_px"]) < 5e-3, (h, residuals[h])
        assert abs(residuals[h]["raw"] - want["raw_px"]) < 5e-3, (h, residuals[h])
    # the whole point: one small drift in both groups for the mapping, a 23% frame error
    # for the alternative, and only at the resolution that discriminates them
    assert abs(residuals[6656]["mapped"] - residuals[8192]["mapped"]) < 1.0
    assert residuals[8192]["raw"] > 100.0


def test_era_pixel_residuals_reads_the_frame_the_slow_test_reads():
    """`gt.era_pixel_residuals` on a synthetic frame with one known answer per height.

    Cheap counterpart to the RUN_SLOW test: it fixes the shape of the emitted block and
    the sign convention, so a refactor of the emitter cannot quietly change what §3's
    numbers mean while the expensive test is skipped.
    """
    df = pd.DataFrame({
        "pano_height": pd.array([6656] * 3 + [8192] * 3 + [1664], dtype="Int64"),
        # rows at raw offsets of 100, 200, 300 px below the horizon in each group
        "current_pano_y": [3228.0, 3128.0, 3028.0, 3996.0, 3896.0, 3796.0, 732.0],
        # stored offset = (h/2 - pano_y) * 6656/h, plus a constant 10 px of drift
        "sv_image_y": [110.0, 210.0, 310.0, 91.25, 172.5, 253.75, 410.0],
    })
    out = gt.era_pixel_residuals(df)
    assert out["n_rows"] == 6                       # the 1664-px row is not a checked group
    assert set(out["by_pano_height"]) == {"6656", "8192"}
    for h in ("6656", "8192"):
        assert out["by_pano_height"][h]["n"] == 3
        assert abs(out["by_pano_height"][h]["mapped_px"] - 10.0) < 1e-9
    # degenerate at the calibration height, and 1.23x wrong at 8192
    assert abs(out["by_pano_height"]["6656"]["raw_px"] - 10.0) < 1e-9
    assert abs(out["by_pano_height"]["8192"]["raw_px"] - (-27.5)) < 1e-9


# ------------------------------------------------------------------ the frozen-model guard

def _ceiling_stub(offsets: dict | None = None) -> tuple[dict, dict]:
    """A committed-ceiling stub and a refit matrix that differs from it by `offsets` m."""
    metrics = {"latlng_median_m": 2.0, "latlng_p90_m": 8.0,
               "dist_median_m": 1.5, "dist_p90_m": 6.0}
    keys = ["gbm_l1", "gbm_dep_l1"]
    ceiling = {"matrix": {k: dict(metrics) for k in keys},
               "meta": {"best_iterations": {k: 100 for k in keys}}}
    era_matrix = {k: dict(metrics) for k in keys}
    for dotted, delta in (offsets or {}).items():
        key, metric = dotted.split(".", 1)
        era_matrix[key][metric] += delta
    return era_matrix, ceiling


def test_the_guard_passes_an_exact_refit_and_calls_it_bit_identical():
    era_matrix, ceiling = _ceiling_stub()
    v = gt.compare_to_committed_ceiling(era_matrix, ceiling, {"gbm_l1": 100, "gbm_dep_l1": 100})
    assert v["within_tolerance"] and v["bit_identical"] and v["best_iterations_match"]
    assert v["exceeded"] == {}


def test_the_guard_accepts_the_measured_cross_host_divergence():
    """The macOS numbers from issue #22, which the shipped 1e-9 assertion rejected.

    This is the regression test for the whole change: gbm_dep_l1 landed 5.9e-5 m from the
    committed era-test median and 3.4e-3 m from its p90 on an Apple-silicon host — four
    orders of magnitude below the 0.40 m ceiling this artifact exists to measure, and an
    order below its tightest published bootstrap bound. It must run, and it must be
    recorded as not bit-identical rather than silently blessed.
    """
    era_matrix, ceiling = _ceiling_stub({"gbm_dep_l1.dist_median_m": 5.9e-5,
                                         "gbm_dep_l1.dist_p90_m": -3.4e-3,
                                         "gbm_dep_l1.latlng_median_m": -3.5e-5})
    v = gt.compare_to_committed_ceiling(era_matrix, ceiling)
    assert v["within_tolerance"] is True
    assert v["bit_identical"] is False
    assert v["worst_metric"] == "gbm_dep_l1.dist_p90_m"
    assert abs(v["max_abs_delta_m"] - 3.4e-3) < 1e-12


@pytest.mark.parametrize("dotted,delta", [
    ("gbm_l1.dist_median_m", 0.05),      # 5 cm on a median: bigger than the modern gaps
    ("gbm_dep_l1.latlng_median_m", -0.2),
    ("gbm_l1.dist_p90_m", 0.5),          # half a metre on a p90
])
def test_the_guard_still_stops_a_genuinely_different_model(dotted, delta):
    """The tolerance is not a licence. Anything that could move a published comparison
    fails, and the failing metric is named so the reader is not left guessing."""
    era_matrix, ceiling = _ceiling_stub({dotted: delta})
    v = gt.compare_to_committed_ceiling(era_matrix, ceiling)
    assert v["within_tolerance"] is False
    assert list(v["exceeded"]) == [dotted]
    assert v["bit_identical"] is False


MEASURED_CROSS_HOST_DELTA_M = 3.4e-3   # issue #22: gbm_dep_l1's era-test p90 on macOS
HEADLINE_EFFECT_M = 0.40               # the ceiling this artifact exists to measure
TIGHTEST_PUBLISHED_BOUND_M = 0.014     # the narrowest paired bootstrap interval it quotes


def test_the_guard_tolerance_sits_between_float_noise_and_the_effect():
    """The design only works as a sandwich, so the sandwich is what is locked.

    Wide enough to admit the largest cross-host divergence anyone has measured, narrow
    enough that nothing it admits could move a number either report leans on. Medians and
    p90s are held to different bounds on purpose: a p90 is an order statistic on a heavy
    tail and moves by the gap between two adjacent rows when a single row changes side,
    while a median over 79,029 era-test rows does not — which is exactly the asymmetry the
    macOS measurement showed (5.9e-5 m on the median, 3.4e-3 m on the p90 of the same fit).
    """
    medians = [gt.CEILING_TOL_M[k] for k in ("dist_median_m", "latlng_median_m")]
    p90s = [gt.CEILING_TOL_M[k] for k in ("dist_p90_m", "latlng_p90_m")]
    assert max(p90s) >= 5 * MEASURED_CROSS_HOST_DELTA_M       # admits the measured case
    assert max(p90s) <= HEADLINE_EFFECT_M / 20                # cannot hide the effect
    assert max(medians) <= TIGHTEST_PUBLISHED_BOUND_M / 10    # nor the tightest bound
    assert max(medians) < min(p90s)
    assert gt.BIT_IDENTICAL_TOL_M < min(gt.CEILING_TOL_M.values())


def test_the_guard_reports_a_round_count_change_without_gating_on_it():
    """A different early-stopping round IS a different booster, so it is recorded — but the
    metrics are what the gate is about, and a run that matches them is still this artifact."""
    era_matrix, ceiling = _ceiling_stub()
    v = gt.compare_to_committed_ceiling(era_matrix, ceiling, {"gbm_l1": 101, "gbm_dep_l1": 100})
    assert v["best_iterations_match"] is False
    assert v["within_tolerance"] is True
    assert v["best_iterations_refit_vs_committed"]["gbm_l1"] == [101, 100]


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
