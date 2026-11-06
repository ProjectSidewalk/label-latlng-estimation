"""The html4-era DC rows (reports/2026-11-05-html4-era-dc-rows.md), locked.

The fast tests assert what the 2026-11-05 run measured, reading the committed
data/html4-era-summary.json only. The summary regenerates offline with
`python python/run_html4_era.py --write`, and one test re-derives the whole summary in-process
so the committed JSON cannot drift from the code.

Headline findings:

- Every DC row with label_id < 66003 holds a linear sv_image_x (the flat html4 renderer's
  formula): 99.99% replay within 1 px under it and 0.4% under the era client's perspective
  replica. Later DC rows are the reverse (0.5% linear, 99.8% era replica).
- Scored against the stored lat/lng, the perspective click bearing is off by 3.1 deg median on
  the html4 rows against 0.9 deg for the linear one; later DC rows are the reverse (1.1 vs 3.0).
- The two models differ by 3.2 deg in bearing at the median row and 5.1 deg at p90, zero at the
  canvas centre, most about 200 px out, less again at the edges; the altitude gap pulls the
  implied camera height from 2.77 m to 2.45 m.
"""

import json
import os

import pytest

from html4_era import summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(ROOT, "data", "html4-era-summary.json")


@pytest.fixture(scope="module")
def summary():
    with open(SUMMARY, encoding="utf-8") as f:
        return json.load(f)


def test_html4_rows_are_linear_and_later_rows_are_era_perspective(summary):
    assert summary["first_perspective_label_id"] == 66003
    assert summary["html4"]["stored_sv_image_x_within_1px_of_linear"] > 0.999
    assert summary["html4"]["stored_sv_image_x_within_1px_of_era_replica"] < 0.01
    assert summary["later"]["stored_sv_image_x_within_1px_of_linear"] < 0.01
    assert summary["later"]["stored_sv_image_x_within_1px_of_era_replica"] > 0.99


def test_formula_classifier_agrees_with_the_id_rule(summary):
    html4 = summary["html4"]["stored_formula_counts"]
    later = summary["later"]["stored_formula_counts"]
    assert html4["era"] == 0 and html4["neither"] < 10
    assert later["linear"] < 100 and later["neither"] < 500


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


def test_model_gap_peaks_off_centre_not_at_the_edges(summary):
    gap = summary["html4"]["model_gap_deg"]["bearing"]
    assert 2.5 < gap["p50"] < 4 and 4 < gap["p90"] < 6 and gap["p100"] > 10
    html4 = summary["html4"]
    by_x = [b["median_deg"] for b in html4["bearing_signed_residual_vs_perspective_by_canvas_x_zoom1"]]
    assert by_x[1] > by_x[0] > 3 and by_x[-2] < by_x[-1] < -2
    by_y = [b["median_deg"] for b in html4["altitude_linear_minus_perspective_by_canvas_y_zoom1"]]
    assert by_y[0] < -3 and by_y[-1] > 2
    heights = summary["html4"]["implied_camera_height_m_median"]
    assert heights["perspective"] < heights["linear"] - 0.25


def _assert_close(fresh, committed, path="summary"):
    if isinstance(committed, dict):
        assert set(fresh) == set(committed), path
        for key in committed:
            _assert_close(fresh[key], committed[key], f"{path}.{key}")
    elif isinstance(committed, list):
        assert len(fresh) == len(committed), path
        for i, (a, b) in enumerate(zip(fresh, committed)):
            _assert_close(a, b, f"{path}[{i}]")
    elif isinstance(committed, float):
        assert fresh == pytest.approx(committed, rel=1e-6, abs=1e-9), path
    else:
        assert fresh == committed, path


def test_summary_rederives_from_the_committed_data(summary):
    _assert_close(summarize(os.path.join(ROOT, "data")), summary)
