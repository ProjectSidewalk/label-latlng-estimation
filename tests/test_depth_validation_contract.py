"""Layer 2 for the issue #9 depth validation: the artifacts' shape, and the offline promise.

The promise these guard is that the analysis outlives the endpoints. Every image-derived
number in the reports comes from `depth-validation-tiles.jsonl.gz`, which holds the
panorama tiles exactly as Google served them, so `build` / `figures` / `gallery` /
`verify_depth_conventions.py` all run with the fetch cache deleted and no network. If
that ever quietly stops being true, the overlay evidence stops being checkable.
"""

import base64
import gzip
import io
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(ROOT, "python"))

TILES = os.path.join(DATA, "depth-validation-tiles.jsonl.gz")

pytestmark = pytest.mark.skipif(
    not os.path.exists(TILES), reason="depth-validation artifacts not built yet"
)


@pytest.fixture(scope="module")
def tiles():
    records = []
    with gzip.open(TILES, "rt", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def panos():
    return pd.read_csv(os.path.join(DATA, "depth-validation-panos.csv.gz"))


def test_tiles_artifact_is_self_describing(tiles):
    assert len(tiles) >= 60
    for rec in tiles:
        assert rec["set"] in {"scoring", "adjudication"}
        assert rec["width"] > 0 and rec["height"] > 0
        assert rec["width"] == 2 * rec["height"], "equirectangular panoramas are 2:1"
        assert rec["tiles"], "a record with no tiles cannot be stitched"


def test_committed_tiles_are_real_jpegs_that_stitch(tiles):
    """The bytes must actually decode -- a truncated artifact would fail silently later."""
    from PIL import Image

    import depth_validation as dv

    rec = tiles[0]
    decoded = [
        {"x": t["x"], "y": t["y"], "bytes": base64.b64decode(t["b64"])}
        for t in rec["tiles"]
    ]
    for tile in decoded:
        with Image.open(io.BytesIO(tile["bytes"])) as img:
            assert img.format == "JPEG"
    rgb = dv.stitch_tiles(decoded, rec["width"], rec["height"])
    assert rgb.shape == (rec["height"], rec["width"], 3)


def test_every_scored_panorama_has_committed_imagery(tiles, panos):
    """No score may depend on a byte that is not in the repo."""
    have = {rec["pano_id"] for rec in tiles if rec["set"] == "scoring"}
    scored = set(panos.loc[panos["has_imagery"].fillna(False), "pano_id"])
    assert scored <= have


def test_panometa_covers_every_payload(tiles):
    """The yaw table must cover every panorama the conventions checks touch."""
    meta = pd.read_csv(os.path.join(DATA, "depth-validation-panometa.csv.gz"))
    assert len(meta) >= 400
    assert meta["yaw_deg"].between(0, 360).all()
    with gzip.open(os.path.join(DATA, "depth-pilot-payloads.jsonl.gz"), "rt",
                   encoding="utf-8") as f:
        payload_ids = {json.loads(line)["pano_id"] for line in f}
    missing = payload_ids - set(meta["pano_id"])
    # A pano can lose its metadata between fetches; a handful is fine, a rift is not.
    assert len(missing) < 0.05 * len(payload_ids)


def test_registration_cohort_is_declared_not_implied(panos):
    """`has_power` records which panoramas could testify at all, rather than hiding it."""
    assert "structure_fraction" in panos.columns
    assert "has_power" in panos.columns
    scored = panos[panos["has_imagery"].fillna(False)]
    assert len(scored) >= 55
    assert scored["structure_fraction"].between(0, 1).all()


def test_adjudication_records_its_own_correction():
    """A hand judgement that was revised must say so, not be quietly overwritten."""
    with open(os.path.join(DATA, "depth-validation-adjudication.json"), encoding="utf-8") as f:
        adj = json.load(f)
    assert adj["sample_size"] == len(adj["verdicts"])
    assert adj["occluded"] == sum(v["occluded"] for v in adj["verdicts"])
    assert "_correction" in adj and adj["_correction"]
    assert "_reviewer" in adj, "a human judgement needs an attributable reviewer"


def test_label_pixel_mapping_requires_the_yaw():
    """Regression guard for the frame bug: the raster is heading-centred.

    Placing a label with the north-referenced sv_image_x alone put markers up to half a
    panorama away from their features. The signature is that the column must move with
    the yaw -- and must not move for a panorama facing due south, where the two frames
    coincide.
    """
    import depth_validation as dv

    width, height = 1664, 832
    sv_x, sv_y = 9026, -200
    south = dv.label_pixel_in_image(sv_x, sv_y, width, height, 180.0)
    rotated = dv.label_pixel_in_image(sv_x, sv_y, width, height, 101.5)
    assert south[0] == pytest.approx(int(sv_x / 13312 * width), abs=2)
    shift = (rotated[0] - south[0]) % width
    assert shift == pytest.approx(int((180.0 - 101.5) / 360.0 * width), abs=2)
    assert rotated[1] == south[1], "the vertical mapping must not depend on yaw"
