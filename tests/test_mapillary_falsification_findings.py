"""The issue #3 Stage 3 census findings, locked.

Convention (mirrors test_pov_inversion_findings.py): the fast tests assert what the
2026-08-07 census measured, reading the committed data/falsification-summary.json only.
The summary regenerates offline and deterministically with
`python python/run_mapillary_falsification.py --write`, and one session-scoped test below
re-derives a run's census in-process so the committed JSON cannot drift from the code.

Headline findings:

- Projection is not a confound: every Mapillary pano in both cities is a true 2:1
  equirect (`spherical`/`equirectangular`), so the falsification never has to model
  perspective or fisheye imagery.
- The pose that matters is the SfM `computed_*` pose. Clovis's raw compass is literal
  zero on 56% of panos, and richmond's SfM moves positions a median ~3 m (p90 ~10 m) —
  scoring against raw EXIF pose would be scoring GPS noise.
- The rig zoo is real but car-shaped: richmond mixes six pano heights from four rigs led
  by a professional car mount (iSTAR Pulsar, 11000x5500, ~69% of site members);
  clovis is a single creator driving one GoPro Fusion at 5760x2880 for two years.
- On-foot capture — the rig class that breaks a car-height cotangent hardest — is
  negligible in both cities (17 / 8,098 and 6 / 8,626 site members), so per-sequence
  camera height only has to separate car-class rigs, not pedestrians.
- The GSV controls place their site members overwhelmingly on 8192-px panos, with a
  6656-px minority — the same two heights the refit was fit on.

Diagnostics findings (the two scale-free axes of #4765/#4766, all six runs):

- The implementation validates externally: where the metric cannot depend on the run's
  rig mix (B and C on the shared-height cities), the range slopes land within a few
  thousandths of #4766's published numbers — including its counterintuitive finding
  that height-normalization alone makes the GSV range axis *worse*.
- The shipped blend D is the flattest model on the range axis on every run: |slope|
  <= 0.09 m/m everywhere, versus the status quo's -0.11..-0.32 on GSV and a
  catastrophic -1.40 on clovis's 2880-px panos (the #4765 sign-flip, measured).
- On the height-residual axis only A reads pixels, so only A can carry a pixel-frame
  height defect — and it does: -0.690 on richmond, 2.6x outside the band the three
  height-blind placements (B, C, D) show from rig confounding alone. That band is a
  region, not a floor: C and D sit well outside B on three of four GSV controls.
- Per-sequence camera heights separate by rig exactly as mount geometry predicts:
  GoPro Max sequences ~13% below the run mean, the iSTAR Pulsar car rig slightly
  above; clovis's two model-strings are one physical rig (both k ~= 0.99).
- And they TRANSFER: scales fitted on a random half of richmond's sites remove ~69%
  of the held-out half's height slope (66-75% over five seeds), against the 91% the
  in-sample fit reports. The in-sample collapse alone proves nothing — pano height is
  constant within a sequence, so one free parameter per sequence absorbs any
  height-correlated systematic by construction. Out-of-sample RMS/range is flat, so
  what the scales carry is height-axis information, not general self-consistency;
  clovis (one physical rig) correctly transfers nothing at all.
"""

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_PATH = os.path.join(ROOT, "data", "falsification-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(SUMMARY_PATH), reason="falsification summary not built yet"
)


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def mapillary(summary):
    return summary["census"]["mapillary"]


# ------------------------------------------------------------------ projection

def test_every_mapillary_pano_is_a_true_equirect(mapillary):
    """No perspective/fisheye imagery slipped through the auto-labeler's is_pano filter."""
    for run, c in mapillary.items():
        assert c["all_true_equirect"], run
        assert set(c["camera_type"]) <= {"spherical", "equirectangular"}, run


def test_richmond_spans_six_pano_heights_clovis_one(mapillary):
    """The 2.8x extrapolation the falsification exists to test (issue #3, Stage 3)."""
    assert sorted(mapillary["richmond"]["pano_heights"]) == [
        "2048", "2688", "2880", "2944", "5500", "6144"]
    assert sorted(mapillary["clovis"]["pano_heights"]) == ["2880"]


# ------------------------------------------------------------------ pose usability

def test_clovis_raw_compass_is_majority_dead(mapillary):
    """56% literal-zero compass_angle: any scoring must use the SfM computed_* pose."""
    c = mapillary["clovis"]
    assert c["raw_field_degeneracy"]["compass_angle_exact_zero"] == 40778
    assert c["n_panos"] == 72776


def test_richmond_sfm_moves_positions_meters(mapillary):
    """Median ~3 m, p90 ~10 m raw-GPS-to-SfM shift; clovis is sub-meter at the median."""
    r = mapillary["richmond"]["sfm_vs_raw"]["position_shift_m"]
    assert 2.5 < r["median"] < 3.5
    assert 8.0 < r["p90"] < 11.0
    assert mapillary["clovis"]["sfm_vs_raw"]["position_shift_m"]["median"] < 0.5


# ------------------------------------------------------------------ the rig zoo

def test_richmond_site_members_are_led_by_the_car_rig(mapillary):
    """iSTAR Pulsar (11000x5500 professional car mount) carries ~69% of site members."""
    rigs = mapillary["richmond"]["rigs"]
    pulsar = rigs["NCTECH LTD / iSTAR Pulsar"]
    total = sum(r["n_site_members"] for r in rigs.values())
    assert total == 8098
    assert pulsar["n_site_members"] == 5614
    assert pulsar["pano_dims"] == {"11000x5500": 4809}


def test_clovis_is_one_creator_one_rig(mapillary):
    c = mapillary["clovis"]
    assert c["n_creators"] == 1
    assert set(c["rigs"]) == {"GoPro / GoPro Fusion FS1.04.01.80.00", "GoPro / Fusion"}


def test_on_foot_capture_is_negligible(mapillary):
    """The rig class that breaks a car-height cotangent hardest barely exists here."""
    for run, member_total in [("richmond", 8098), ("clovis", 8626)]:
        modes = mapillary[run]["capture_modes"]
        assert sum(v["n_site_members"] for v in modes.values()) == member_total, run
        assert modes["on_foot"]["n_site_members"] <= 20, run


# ------------------------------------------------------------------ GSV controls

def test_gsv_controls_sit_on_the_fit_heights(summary):
    """Site members live on 8192 with a 6656 minority — the refit's own training heights."""
    for run, c in summary["census"]["gsv_control"].items():
        by_height = c["pano_heights"]
        members = {h: v["n_site_members"] for h, v in by_height.items()}
        assert max(members, key=members.get) == "8192", run
        assert members.get("1664", 0) < 300, run


# ------------------------------------------------------------------ re-derivation

def test_census_rederives_from_committed_inputs(mapillary):
    """The committed JSON cannot drift from the code: rebuild richmond's census in-process."""
    import mapillary_falsification as mf

    fresh = json.loads(json.dumps(mf.census_mapillary_run("richmond")))
    assert fresh == mapillary["richmond"]


# ------------------------------------------------------------------ diagnostics

@pytest.fixture(scope="module")
def diag(summary):
    return summary["diagnostics"]["runs"]


ALL_RUNS = ["richmond", "clovis", "paterson", "gainesville", "bend", "sao_paulo"]


def test_conventions_pin_exactly(summary):
    """Our cotangent at h=2.6 reproduces the auto-labeler's stored ray ranges."""
    assert summary["diagnostics"]["conventions"]["max_abs_range_m_delta"] < 0.001


def test_range_slopes_validate_against_4766(diag):
    """Where the metric cannot depend on rig mix, we land on #4766's published numbers."""
    published = {  # SidewalkWebpage#4766's table, range slope column
        ("paterson", "C_cotangent"): 0.0983,
        ("paterson", "B_normalized"): -0.4496,
        ("richmond", "C_cotangent"): 0.1207,
        ("richmond", "B_normalized"): -0.2901,
    }
    for (run, model), theirs in published.items():
        ours = diag[run]["per_model"][model]["range_slope"]["slope"]
        assert abs(ours - theirs) < 0.02, (run, model, ours, theirs)


def test_normalization_alone_worsens_the_gsv_range_axis(diag):
    """#4766's counterintuitive finding, reproduced on all four GSV controls."""
    for run in ["paterson", "gainesville", "bend", "sao_paulo"]:
        a = diag[run]["per_model"]["A_status_quo"]["range_slope"]["slope"]
        b = diag[run]["per_model"]["B_normalized"]["range_slope"]["slope"]
        assert b < a < 0, run


def test_blend_range_axis_is_flat_everywhere(diag):
    """The falsification's first axis: no compression signature on any run. D beats the
    linear models everywhere and the raw cotangent on five of six runs; the exception is
    sao_paulo (+0.030 for C vs -0.090 for D), whose far-heavy member mix sits in D's
    linear tail — the designed near-horizon trade, visible exactly where it should be."""
    for run in ALL_RUNS:
        per_model = diag[run]["per_model"]
        d = abs(per_model["D_blend"]["range_slope"]["slope"])
        assert d <= 0.091, run
        for other in ["A_status_quo", "B_normalized"]:
            assert d < abs(per_model[other]["range_slope"]["slope"]), (run, other)
        beats_cot = d < abs(per_model["C_cotangent"]["range_slope"]["slope"])
        assert beats_cot == (run != "sao_paulo"), run


def test_clovis_status_quo_compression_is_catastrophic(diag):
    """The #4765 sign-flip measured: raw pixels on a uniform 2880-px city."""
    assert diag["clovis"]["per_model"]["A_status_quo"]["range_slope"]["slope"] < -1.0


def test_only_the_pixel_reading_model_leaves_a_height_defect(diag):
    """The falsification's second axis. B, C and D take no pano_height input, so their
    slopes bound the rig-confounding band; A is the only candidate that reads pixels and
    the only one that can carry a pixel-frame defect. On richmond it sits 2.6x outside
    the band — #4765's defect, alive on Mapillary."""
    per_model = diag["richmond"]["per_model"]
    band = max(abs(per_model[k]["height_slope"]["slope"])
               for k in ["B_normalized", "C_cotangent", "D_blend"])
    assert abs(per_model["D_blend"]["height_slope"]["slope"]) <= band
    assert abs(per_model["A_status_quo"]["height_slope"]["slope"]) > 2 * band
    # and A points the way the mechanism predicts on every run that has two pano heights
    for run in ["richmond", "paterson", "gainesville", "bend", "sao_paulo"]:
        assert diag[run]["per_model"]["A_status_quo"]["height_slope"]["slope"] < 0, run


def test_the_height_blind_band_is_a_region_not_a_floor(diag):
    """B is not a universal confound floor: on three of four GSV controls both C and D
    sit well outside it, which is why the report reads the three as a band. Locked so the
    'D is at the floor' reading cannot come back by accident."""
    for run in ["paterson", "gainesville", "bend"]:
        per_model = diag[run]["per_model"]
        b = abs(per_model["B_normalized"]["height_slope"]["slope"])
        for other in ["C_cotangent", "D_blend"]:
            assert abs(per_model[other]["height_slope"]["slope"]) > b + 0.05, (run, other)


def test_single_height_runs_report_no_height_slope(diag):
    """Clovis has one pano height; a numeric slope there would be float noise."""
    for model, v in diag["clovis"]["per_model"].items():
        assert v["height_slope"]["slope"] is None, model


# ------------------------------------------------------------------ sequence scales

@pytest.fixture(scope="module")
def scales(summary):
    return summary["sequence_scales"]


def test_per_sequence_heights_collapse_richmonds_height_slope(scales):
    """In-sample: fitted camera heights take the height slope to ~0. On its own this
    proves nothing (see the holdout tests below) — pano height is constant within a
    sequence, so one free parameter per sequence absorbs it by construction."""
    r = scales["richmond"]
    assert abs(r["d_blend_unscaled"]["height_slope"]["slope"]) > 0.2
    assert abs(r["d_blend_per_sequence_scale"]["height_slope"]["slope"]) < 0.05
    # and the range axis stays flat rather than being traded away
    assert abs(r["d_blend_per_sequence_scale"]["range_slope"]["slope"]) < 0.02


def test_single_sequence_sites_are_excluded_from_the_objective(scales):
    """A site seen by one sequence scales with that sequence, so it pulls k toward zero
    with nothing opposing it — zero information about relative scale, pure degeneracy.
    Only multi-sequence members enter the fit."""
    for run, total in [("richmond", 7711), ("clovis", 7691)]:
        assert 0 < scales[run]["n_members_in_objective"] < total, run
    assert scales["richmond"]["n_members_in_objective"] == 7186   # 525 members dropped
    assert scales["clovis"]["n_members_in_objective"] == 5631     # 2,060 dropped (27%)


def test_fitted_rig_heights_transfer_to_held_out_sites(scales):
    """The finding the in-sample collapse cannot deliver: scales fitted on a random half
    of richmond's sites remove ~69% of the DISJOINT half's height slope. Real rig
    geometry, measured with no ground truth."""
    h = scales["richmond"]["holdout"]
    before = abs(h["d_blend_unscaled"]["height_slope"]["slope"])
    after = abs(h["d_blend_transferred_scale"]["height_slope"]["slope"])
    assert before > 0.2
    assert after < 0.4 * before          # ~69% removed
    assert h["n_held_members"] > 3000 and h["held_members_covered"] > 3000


def test_the_transfer_is_not_a_lucky_split(scales):
    """Same result on five seeds: 66-75% of the held-out slope removed, every time."""
    sweep = scales["richmond"]["holdout_seed_sweep"]
    assert len(sweep) == 5
    removed = [1 - abs(w["d_blend_transferred_scale"]["height_slope"]["slope"])
               / abs(w["d_blend_unscaled"]["height_slope"]["slope"]) for w in sweep]
    assert min(removed) > 0.6, removed
    assert max(removed) < 0.85, removed


def test_the_transfer_buys_the_height_axis_and_not_self_consistency(scales):
    """RMS/range improves 6% in-sample but is flat out-of-sample, so the fitted scales
    carry height-axis information specifically — the in-sample RMS gain was the added
    parameters. Stated so the report cannot quietly claim more than that."""
    r = scales["richmond"]
    in_sample = (r["d_blend_unscaled"]["rms_over_range"]
                 - r["d_blend_per_sequence_scale"]["rms_over_range"])
    h = r["holdout"]
    held = h["d_blend_unscaled"]["rms_over_range"] - h["d_blend_transferred_scale"]["rms_over_range"]
    assert in_sample > 0.008
    assert abs(held) < 0.002


def test_clovis_one_rig_transfers_nothing(scales):
    """The null control: one physical camera, so there is no per-sequence height to find,
    and transferring the fitted scales to held-out sites does not help (it slightly
    hurts). A method that only fit noise could not produce this."""
    h = scales["clovis"]["holdout"]
    assert (h["d_blend_transferred_scale"]["rms_over_range"]
            > h["d_blend_unscaled"]["rms_over_range"])


def test_rig_classes_separate_as_mount_geometry_predicts(scales):
    rigs = scales["richmond"]["per_rig"]
    gopro = rigs["GoPro / GoPro Max"]["k_rel_median"]
    pulsar = rigs["NCTECH LTD / iSTAR Pulsar"]["k_rel_median"]
    assert gopro < 0.93 < pulsar < 1.10


def test_clovis_is_one_physical_rig(scales):
    """Two Mapillary model-strings, one camera: both fit within 2% of the run mean."""
    for rig, r in scales["clovis"]["per_rig"].items():
        assert abs(r["k_rel_median"] - 1.0) < 0.02, rig


def test_relative_scale_is_identified(scales):
    """Most sites see more than one sequence, so relative rig scale is measured, not
    assumed (the global scale stays unidentified — RampNet#101)."""
    assert scales["richmond"]["n_multi_sequence_sites"] > 900
    assert scales["clovis"]["n_multi_sequence_sites"] > 900
