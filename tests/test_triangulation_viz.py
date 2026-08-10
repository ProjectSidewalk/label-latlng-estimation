"""The committed conclusions page (issue #7 viz) stays in lock-step with the summary.

The page is context, not evidence — no report number depends on it — but once committed
it must not silently drift from ``data/triangulation-summary.json``, and it must stay
self-contained (every pixel embedded, nothing fetched at view time).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
PAGE = os.path.join(ROOT, "figures", "triangulation-conclusions.html")
SUMMARY = os.path.join(ROOT, "data", "triangulation-summary.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(PAGE), reason="conclusions page not built yet"
)


@pytest.fixture(scope="module")
def page_data():
    with open(PAGE, encoding="utf-8") as f:
        html = f.read()
    m = re.search(r"const DATA = (\{.*?\});\n", html, re.S)
    assert m, "page carries no embedded DATA blob"
    return json.loads(m.group(1))


def test_page_numbers_match_the_committed_summary(page_data):
    with open(SUMMARY, encoding="utf-8") as f:
        s = json.load(f)
    ps = page_data["summary"]
    assert ps["pooled_ratio"] == s["depth_anchor"]["pooled"]["median_ratio_tri_over_depth"]
    assert ps["shipped"] == s["meta"]["shipped_height_m"]
    for run, v in ps["scale_global"].items():
        assert v["height_m"] == s["scale_global"][run]["height_m"], run
    assert ps["gap_range_profile"] == s["depth_anchor"]["gap_range_profile"]


def test_page_is_self_contained_with_real_pixels(page_data):
    sites = page_data["sites"]
    assert len(sites) == 6
    for site in sites:
        assert len(site["members"]) >= 4, site["site"]
        for m in site["members"]:
            assert m["img"].startswith("data:image/jpeg;base64,"), site["site"]
            # the viz top-up bundle gives every camera a depth-model panel
            assert m.get("dimg", "").startswith("data:image/png;base64,"), site["site"]
        # ...but committed r_depth values remain exactly the §8 anchor population
        assert (sum(1 for m in site["members"] if m["rd"] is not None)
                == site["n_depth"]), site["site"]


def test_every_site_stands_on_its_own_georeferenced_ground(page_data):
    """The aerial patch is cut for exactly the +/-ext metres the scene draws.

    Nothing on screen would look wrong if the two disagreed — the imagery would simply
    sit at the wrong scale under geometry that is still drawn correctly, which is the
    kind of error a reader would read straight past. So the extent travels with the
    patch and is asserted here as well as refused at build time.
    """
    from PIL import Image

    for site in page_data["sites"]:
        g = site.get("ground")
        assert g is not None, f"site {site['site']} has no aerial ground"
        assert g["img"].startswith("data:image/jpeg;base64,"), site["site"]
        ext = site["ext"]
        assert isinstance(ext, int) and ext >= 12 and ext % 2 == 0, site["site"]
        # every camera has to stand on the ground that was fetched for it
        far = max(max(abs(m["e"]), abs(m["n"])) for m in site["members"])
        assert ext >= far, (f"site {site['site']}: ground reaches {ext} m but a camera "
                            f"stands at {far:.1f} m")
        patch = Image.open(io.BytesIO(base64.b64decode(g["img"].split(",", 1)[1])))
        assert patch.width == patch.height, f"site {site['site']}: patch is not square"


def test_page_declares_a_doctype_and_charset():
    """Committed as a standalone artifact, the page must not render in quirks mode."""
    with open(PAGE, encoding="utf-8") as f:
        head = f.read(200).lower()
    assert head.startswith("<!doctype html>")
    assert 'charset="utf-8"' in head


def test_zoom3_dimensions_follow_the_pano_generation():
    """The zoom-3 pyramid is native/4 — hardcoding 4096x2048 put two wrong crops on the
    committed page (gen-3 panos serve 3328x1664). Locked at the unit level."""
    import triangulation_viz as tv

    assert tv.pano_zoom3_wh(8192) == (4096, 2048)
    assert tv.pano_zoom3_wh(6656) == (3328, 1664)
    # and the derivation agrees with what Google itself reported for every anchor pano
    sizes = tv._anchor_image_wh()
    assert len(sizes) >= 400
    assert all(w == 2 * h for w, h in sizes.values())
    assert {(3328, 1664), (4096, 2048)} >= set(sizes.values())


def test_crop_tiles_wrap_at_the_true_pixel_seam():
    """Tile ids must wrap at the served width, not at a tile multiple: gen-3's 3328 px
    is 6.5 tiles, so the seam falls mid-tile and `col % (w // TILE)` names tiles that do
    not exist (review's finding, generalised)."""
    import triangulation_viz as tv

    w, h = 3328, 1664
    ids = tv._crop_tiles(0.99, w, h)                 # crosses the seam
    cols = {c for c, _ in ids}
    assert max(cols) <= (w - 1) // tv.TILE, cols     # never a tile past the last real one
    assert 0 in cols and 6 in cols                   # both sides of the seam covered
    rows = {r for _, r in ids}
    assert rows == {1, 2}                            # the crop band of a 1664-row pano


def test_crop_image_samples_across_a_mid_tile_seam():
    """Pixel-exact seam handling on synthetic tiles: a window crossing gen-3's half-tile
    seam must read column (x mod w), never black padding or a misplaced tile."""
    from PIL import Image
    import triangulation_viz as tv

    w, h = 3328, 1664
    tiles = {}
    for cx in range(7):
        for cy in range(1, 3):
            a = np.zeros((512, 512, 3), np.uint8)
            a[:, :, 0] = cx * 30 + 20                # encode the column id in red
            tiles[(cx, cy)] = _png_bytes(Image.fromarray(a))
    img = np.asarray(tv.crop_image(tiles, 0.995, w, h))
    reds = set(np.unique(img[:, :, 0]))
    # the window spans the seam: last-column tile (6) and first tiles (0, 1) — never the
    # black (0) a padding read would produce
    assert 0 not in reds
    assert {20, 200}.issubset(reds), reds


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_committed_tile_bundle_matches_each_panos_true_grid():
    """The bundle must hold tiles only where the pano's own pyramid has content — the
    stale bundle carried all-black nadir tiles for the gen-3 pano's nonexistent rows."""
    import gzip

    import triangulation_viz as tv

    sizes = tv._anchor_image_wh()
    bundle = os.path.join(ROOT, "data", "triangulation-viz-tiles.jsonl.gz")
    with gzip.open(bundle, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            wh = sizes.get(rec["pano_id"])
            if wh is None:
                continue
            w, hh = wh
            max_col = (w - 1) // tv.TILE
            max_row = (int(tv.ROW_HI * hh) - 1) // tv.TILE
            for t in rec["tiles"]:
                assert t["x"] <= max_col, (rec["pano_id"], t["x"])
                assert t["y"] <= max_row, (rec["pano_id"], t["y"])


@pytest.mark.skipif(not os.environ.get("RUN_SLOW"),
                    reason="full offline page rebuild (~8 min); set RUN_SLOW=1")
def test_page_rebuilds_byte_identically_from_committed_bundles(tmp_path, monkeypatch):
    """The MANIFEST's reproducibility claim, locked: `build` from a fresh checkout must
    reproduce the committed page exactly. Byte-identity is a property of the pinned
    environment (PIL's encoders) — a mismatch after a Pillow upgrade means re-verify and
    recommit, not that the page lied."""
    import triangulation_viz as tv

    out = tmp_path / "page.html"
    monkeypatch.setattr(tv, "OUT_HTML", out)
    tv.build()
    with open(PAGE, encoding="utf-8") as f:
        committed = f.read()
    assert out.read_text(encoding="utf-8") == committed


def test_depth_panel_covers_the_same_window_shape_as_the_photo(page_data):
    """The two panels claim to be the identical window, so they must share an aspect.

    They are rendered at different pixel widths on purpose — the depth raster is coarse
    and gains nothing from upsampling — but the page lays both out at one CSS width, so
    any aspect disagreement shows up directly as a stretched, taller-than-the-photo
    depth panel next to it.
    """
    from PIL import Image

    def shape(uri):
        return Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).size

    for site in page_data["sites"]:
        for m in site["members"]:
            pw, ph = shape(m["img"])
            dw, dh = shape(m["dimg"])
            assert abs((dh / dw) / (ph / pw) - 1) < 0.02, (
                f"site {site['site']}: depth panel {dw}x{dh} does not match the "
                f"photo's {pw}x{ph} aspect")
