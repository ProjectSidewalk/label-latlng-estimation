"""The 2021 findings must reproduce on the reconstructed data.

The original 2021 CSVs are gone (issue #1), so this is a tolerance-based comparison against the
frozen published output (scripts/label-latlng-estimation.md), not a bit-for-bit one: the
committed data is a reconstruction from production, and the 2021 train/test RNG draw cannot be
replayed. What must hold, and what these tests assert:

- the cleaning pipeline lands on EXACTLY the published row counts (395,147 / 316,118 / 79,029) —
  the depth-label population turned out to be fully frozen in production, so any deviation means
  the data or pipeline changed;
- estimate 7 wins, with the published ranking (est5/est6 and est2/est3 are published ties —
  1.79/1.79 and 4.63/4.64 m — so each pair may appear in either order);
- the six winning regressions predict within tight absolute bounds of the published coefficients
  (per-coefficient relative comparison is meaningless for the tiny canvas_y terms, so the
  assertion is on predictions over the observed predictor range);
- the headline accuracy (1.47 m median error) reproduces closely.

Tolerances here were set from the observed 2026 reconstruction run (max prediction deviation
~0.6 m at zoom 3; median error delta 0.008 m) with margin; see data/MANIFEST.md.
"""

import numpy as np

# Frozen output of the 2021 analysis (scripts/label-latlng-estimation.md).
PUBLISHED_ROWS = {"cleaned": 395_147, "train": 316_118, "test": 79_029}
PUBLISHED_MEDIAN_ERROR_M = 1.47
PUBLISHED_EST7 = {
    "dist": [  # (Intercept), sv_image_y, canvas_y — zooms 1, 2, 3
        {"(Intercept)": 18.6051843, "sv_image_y": 0.0138947, "canvas_y": 0.0011023},
        {"(Intercept)": 20.8794248, "sv_image_y": 0.0184087, "canvas_y": 0.0022135},
        {"(Intercept)": 25.2472682, "sv_image_y": 0.0264216, "canvas_y": 0.0011071},
    ],
    "heading": [  # (Intercept), canvas_x
        {"(Intercept)": -51.2401711, "canvas_x": 0.1443374},
        {"(Intercept)": -27.5267447, "canvas_x": 0.0784357},
        {"(Intercept)": -13.5675945, "canvas_x": 0.0396061},
    ],
}
PUBLISHED_RANKING = ["est7", {"est5", "est6"}, "est4", {"est2", "est3"}, "est1"]

MAX_DIST_PRED_DEVIATION_M = 1.0
MAX_HEADING_PRED_DEVIATION_DEG = 0.5
MAX_MEDIAN_ERROR_DELTA_M = 0.05


def test_row_counts_reproduce_published(analysis):
    assert analysis["meta"]["rows_after_cleaning"] == PUBLISHED_ROWS["cleaned"]
    assert analysis["meta"]["rows_train"] == PUBLISHED_ROWS["train"]
    assert analysis["meta"]["rows_test"] == PUBLISHED_ROWS["test"]


def test_est7_wins_with_published_ranking(analysis):
    order = [r["estimate"].removeprefix("error_") for r in analysis["error_stats"]["summary"]]
    i = 0
    for slot in PUBLISHED_RANKING:
        if isinstance(slot, set):
            assert set(order[i:i + len(slot)]) == slot, f"ranking differs at {order}"
            i += len(slot)
        else:
            assert order[i] == slot, f"ranking differs at {order}"
            i += 1
    assert i == len(order)


def test_est7_median_error_reproduces(analysis):
    med = next(r["median"] for r in analysis["error_stats"]["summary"]
               if r["estimate"] == "error_est7")
    assert abs(med - PUBLISHED_MEDIAN_ERROR_M) < MAX_MEDIAN_ERROR_DELTA_M, med


def _pred(coefs: dict, **terms) -> np.ndarray:
    out = np.full_like(next(iter(terms.values())), coefs["(Intercept)"], dtype=float)
    for name, values in terms.items():
        out = out + coefs[name] * values
    return out


def test_est7_coefficients_predict_like_published(analysis, raw_data):
    """Refit vs published coefficients, compared where it matters: on predictions over the
    deciles of the observed predictor ranges, per zoom level."""
    from label_latlng_estimation import clean_data

    cleaned, _ = clean_data(raw_data)
    q = np.linspace(0, 1, 11)
    for z in range(3):
        sub = cleaned[cleaned["zoom"] == z + 1]
        svy = sub["sv_image_y"].quantile(q).to_numpy(float)
        cy = sub["canvas_y"].quantile(q).to_numpy(float)
        cx = sub["canvas_x"].quantile(q).to_numpy(float)

        d_now = _pred(analysis["est7"]["dist"][z], sv_image_y=svy, canvas_y=cy)
        d_pub = _pred(PUBLISHED_EST7["dist"][z], sv_image_y=svy, canvas_y=cy)
        assert np.max(np.abs(d_now - d_pub)) < MAX_DIST_PRED_DEVIATION_M, f"zoom {z + 1} dist"

        h_now = _pred(analysis["est7"]["heading"][z], canvas_x=cx)
        h_pub = _pred(PUBLISHED_EST7["heading"][z], canvas_x=cx)
        assert np.max(np.abs(h_now - h_pub)) < MAX_HEADING_PRED_DEVIATION_DEG, f"zoom {z + 1} heading"
