"""Layer 3 for issue #9's coordinate-convention findings: the conclusions, locked.

Convention (mirrors test_depth_pilot_findings.py): these assert what
``python/verify_depth_conventions.py`` measured against the committed artifacts. The
fast tests read the committed evidence JSON only, so on their own they turn red when a
*regenerated* artifact changes a conclusion; a regression in the decoder or the frame
helpers reaches them after rerunning the verify script -- or in-process through the
RUN_SLOW=1 re-derivation test at the bottom of this file, which recomputes every check
from committed bytes and compares dicts.

The findings, and why each matters:

- streetlevel's ``DepthMap`` is the x-mirror of the payload order the 2017-2020 client
  used. Both conventions come from GSVPanoDepth.js -- ``computeDepthMap`` mirrors,
  ``computePointCloud`` does not -- so anything indexing the saved array with a pixel
  coordinate must mirror first. Unmirrored the two differ by a median of ~75 m.
- ``sv_image_x`` is a true compass bearing; the panorama raster and the depth payload
  are heading-centred (column 0 is ``yaw - 180``). Mixing them displaces a label by up
  to half a panorama, and by nothing at all for a panorama facing due south.
- That mismatch moves a stored label position by 0 m at the median but past 3 m for
  ~7% of labels -- small because the depth model is nearly flat earth.

Thresholds are deliberately loose where the underlying quantity is a measurement rather
than an identity, so the suite tracks conclusions and not noise.
"""

import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EVIDENCE = os.path.join(DATA, "depth-conventions-evidence.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(EVIDENCE),
    reason="run `python python/verify_depth_conventions.py --json` first",
)


@pytest.fixture(scope="module")
def evidence():
    with open(EVIDENCE, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- A: decoder orientation

def test_streetlevel_disagrees_wildly_unmirrored(evidence):
    a = evidence["A_decoder_orientation"]
    assert a["payloads"] >= 400
    # Not a rounding difference: whole scenes apart.
    assert a["asis_median_max_diff_m"] > 10.0


def test_streetlevel_matches_after_mirroring(evidence):
    a = evidence["A_decoder_orientation"]
    # Every payload agrees once x is mirrored; the residual is float32 quantization in
    # the v6 replica at long range, not a disagreement about the geometry.
    assert a["agree_after_mirroring_within_10cm"] == a["payloads"]
    assert a["mirrored_worst_max_diff_m"] < 0.1


# ---------------------------------------------------------------- B/C: sv_image_x

def test_sv_image_x_is_a_compass_bearing(evidence):
    b = evidence["B_sv_image_x_frame"]
    assert b["verdict"] == "A"
    assert b["labels"] > 390_000
    compass = b["A_sv_image_x_is_compass_bearing"]
    heading_relative = b["B_sv_image_x_is_heading_relative"]
    # Against the independently recorded POV heading, the compass reading leaves a
    # viewport-sized residual; the heading-relative one smears over the whole circle.
    assert compass["within_60deg_frac"] > 0.98
    assert heading_relative["within_60deg_frac"] < 0.6
    assert compass["std"] < heading_relative["std"] / 3


def test_sv_image_x_reproduces_the_stored_bearing(evidence):
    # True by construction -- the stored position was written from this coordinate --
    # so a drift here means the decode path changed, not that the frame moved.
    assert evidence["B_sv_image_x_frame"]["bearing_vs_sv_image_x_median_deg"] < 2.0


def test_current_pano_x_differs_by_exactly_the_yaw(evidence):
    c = evidence["C_pano_x_vs_sv_image_x"]
    assert c["rows"] > 190_000
    assert c["correlation_with_yaw_offset"] > 0.9


# ---------------------------------------------------------------- D/E: the raster frame

def test_raster_is_heading_centred(evidence):
    d = evidence["D_raster_frame"]
    assert d["links"] > 50
    assert d["verdict"] == "B"
    assert d["B_heading_centred_roadness"] > d["A_north_referenced_roadness"]


def test_payload_shares_the_rasters_frame(evidence):
    # Same conclusion from the depth alone, with no imagery involved.
    e = evidence["E_payload_frame"]
    assert e["links"] > 300
    assert e["verdict"] == "B"
    assert e["B_exceeds_pano_median"] > e["A_exceeds_pano_median"]


# ---------------------------------------------------------------- F: registration

def test_payload_order_registers_against_the_imagery(evidence):
    f = evidence["F_registration"]
    assert f["panoramas"] >= 50
    assert f["payload_order_better_on"] > 3 * f["mirrored_better_on"]
    assert f["payload_order_median_violation"] < f["mirrored_median_violation"]


# ---------------------------------------------------------------- G: what it costs

def test_frame_mismatch_is_small_at_the_median_but_has_a_tail(evidence):
    g = evidence["G_frame_impact"]
    assert g["labels"] > 700
    # Nearly flat earth, so reading the wrong azimuth usually returns the same range...
    assert g["median_shift_m"] < 0.1
    assert g["frac_within_1m"] > 0.75
    # ...but not always, and the tail is the part that matters for a refit.
    assert 0.02 < g["frac_beyond_3m"] < 0.20


# ---------------------------------------------------------------- the evidence itself

@pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="re-derives every check from committed bytes (minutes); set RUN_SLOW=1",
)
def test_run_checks_reproduces_the_committed_evidence(evidence):
    """The committed evidence JSON is exactly what the verify script computes today.

    The bridge the fast locks above deliberately do not provide: they assert the
    committed numbers, and this recomputes those numbers from the committed bytes, so
    a silent regression in the decoder or the frame helpers cannot hide behind a stale
    artifact.
    """
    import sys

    sys.path.insert(0, os.path.join(ROOT, "python"))
    from verify_depth_conventions import run_checks

    assert run_checks() == evidence
