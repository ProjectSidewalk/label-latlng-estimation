"""The html4-era DC rows (reports/2026-11-05-html4-era-dc-rows.md), locked.

The fast tests assert what the 2026-11-05 run measured, reading the committed
data/html4-era-summary.json only. The summary regenerates offline with
`python python/run_html4_era.py --write`, and one session-scoped test re-derives the headline
numbers in-process so the committed JSON cannot drift from the code.

Headline findings:

- Every DC row with label_id < 66003 holds a linear sv_image_x/y (the flat html4 renderer's
  formula), not the perspective inversion: 99.99% of them replay within 1 px under the linear
  formula and 0.01% under pov_if_centered. Later DC rows are the reverse.
- Scored against the stored lat/lng, the perspective click bearing is off by 3.1 deg median on
  those rows against 0.9 deg for the linear one; later DC rows are the reverse (1.1 vs 3.0).
- The error has the projection signature: it grows away from the canvas centre, flips sign
  across it, and pulls the implied camera height from 2.77 m to 2.45 m.
"""

import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "data", "html4-era-summary.json")


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY) as f:
        return json.load(f)


def test_html4_rows_are_linear_and_later_rows_are_not(summary):
    assert summary["first_perspective_label_id"] == 66003
    assert summary["html4"]["stored_sv_image_within_1px_of_linear"] > 0.999
    assert summary["html4"]["stored_sv_image_within_1px_of_perspective"] < 0.01
    assert summary["later"]["stored_sv_image_within_1px_of_linear"] < 0.01


def test_html4_rows_are_a_quarter_of_cleaned_dc(summary):
    assert summary["html4_rows_raw"] == 65815
    assert summary["html4_rows_cleaned"] == 56347
    assert 0.2 < summary["html4_share_of_cleaned_dc"] < 0.3


def test_perspective_bearing_is_wrong_on_html4_rows_and_right_later(summary):
    h = summary["html4"]["bearing_abs_residual_median_deg"]
    later = summary["later"]["bearing_abs_residual_median_deg"]
    assert h["perspective"] > 2.5 * h["linear"]
    assert later["linear"] > 2.5 * later["perspective"]
    assert h["linear"] == pytest.approx(later["perspective"], abs=0.3)


def test_error_has_the_projection_signature(summary):
    by_x = summary["html4"]["bearing_signed_residual_vs_perspective_by_canvas_x_zoom1"]
    assert by_x[0]["median_deg"] > 3 and by_x[-1]["median_deg"] < -2
    by_y = summary["html4"]["altitude_linear_minus_perspective_by_canvas_y_zoom1"]
    assert by_y[0]["median_deg"] < -3 and by_y[-1]["median_deg"] > 2
    heights = summary["html4"]["implied_camera_height_m_median"]
    assert heights["perspective"] < heights["linear"] - 0.25


def test_summary_rederives_from_the_committed_data(summary):
    sys.path.insert(0, os.path.join(ROOT, "python"))
    from html4_era import summarize

    fresh = summarize(os.path.join(ROOT, "data"))
    for group in ("html4", "later"):
        for key in ("stored_sv_image_within_1px_of_linear", "stored_sv_image_within_1px_of_perspective"):
            assert fresh[group][key] == pytest.approx(summary[group][key], abs=1e-9)
        for key, value in summary[group]["bearing_abs_residual_median_deg"].items():
            assert fresh[group]["bearing_abs_residual_median_deg"][key] == pytest.approx(value, rel=1e-6)
