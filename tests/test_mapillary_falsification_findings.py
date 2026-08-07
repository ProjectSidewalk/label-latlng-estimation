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
