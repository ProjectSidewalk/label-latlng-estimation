# Dataset manifest — depth-derived label positions (RECONSTRUCTION)

**These are not the original 2021 CSVs.** The files consumed by the 2021 analysis were never
committed and no longer exist ([issue #1](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/1)).
The `labels-*-latlng.csv.gz` files here were **regenerated from production on 2026-08-05** using
`scripts/extraction/`. That said, the reconstruction turned out to be reproduction-grade: the
depth-label population has been effectively frozen in production since 2021, and re-running the
2021 cleaning pipeline on these files yields **exactly** the published row counts
(395,147 cleaned; 316,118 train / 79,029 test) and reproduces the published findings — see
`tests/` and the R baseline in `tests/fixtures/r-baseline/`.

## Extraction provenance

- **Date:** 2026-08-05 (UTC), host `makelab1.cs.washington.edu`
- **Databases:** `sidewalk_prod` (PostgreSQL 16, schema per city, Play evolutions applied: 340)
  for the six modern cities; `sidewalk_dc` (the retired legacy DC database, still at Play
  evolution 19 with the pre-evolution-179 schema) for DC
- **Queries:** `scripts/extraction/extract-depth-labels-modern.sql` (six cities) and
  `extract-depth-labels-legacy-dc.sql` (DC — near-verbatim the 2021 Rmd's documented DC query);
  raw psql session details in `data/extraction-metadata.txt`
- **Selection:** `computation_method = 'depth'` (modern) / `lat IS NOT NULL` (legacy DC). The
  2021 extraction was unfiltered, but every non-depth row was discarded by the analysis'
  cleaning filters, so nothing the analysis uses is lost.

## Row counts

| city | source | rows | rows with `time_created` ≥ 2021-01-01 |
|---|---|---|---|
| dc | `sidewalk_dc` (legacy schema, evolution 19) | 270,845 | 61 (plus 82,791 NULL — early rows predate the column) |
| seattle | `sidewalk_prod.sidewalk_seattle` | 120,094 | 3,808 |
| newberg | `sidewalk_prod.sidewalk_newberg` | 17,725 | 38 |
| columbus | `sidewalk_prod.sidewalk_columbus` | 20,526 | 395 |
| spgg | `sidewalk_prod.sidewalk_spgg` | 31,498 | 792 |
| cdmx | `sidewalk_prod.sidewalk_cdmx` | 6,772 | 100 |
| pittsburgh | `sidewalk_prod.sidewalk_pittsburgh` | 1,148 | 412 |
| **total** | | **468,608** | |

Verified at extraction time: every depth row in the six modern cities has an
`old_label_metadata` record (0 missing across all cities), so the pre-evolution-179 values of
`sv_image_x/y`, `photographer_heading/pitch`, and `panorama_lat/lng` are recovered exactly.

## Columns

Columns 1–22 are **exactly the 2021 column set** (names, order, and semantics of the original
extraction — see the mapping comments in `scripts/extraction/extract-depth-labels-modern.sql`).
Columns 23–27 are extras that did not exist in the 2021 CSVs, appended so both `readr` and
`pandas` read the files unchanged:

- `pano_width`, `pano_height` — panorama pixel dimensions from today's `pano_data` (pano
  resolution never changes for a given pano id, so these are valid for when the label was
  placed; NULL where GSV no longer serves the pano — e.g. 2,017 Seattle rows)
- `time_created` — label creation time (NULL for early DC rows that predate the column)
- `current_pano_x`, `current_pano_y` — the post-evolution-179 *recomputed* pano coordinates,
  for cross-checking against the original `sv_image_x/y` (NULL for DC: evolution 179 never ran
  on the legacy database)

## Caveats

1. **The `time_created < 2021-01-01` cutoff matters for faithfulness.** Evolution 93 stamped
   `computation_method = 'depth'` on every row that had lat/lng when it ran (mid-2021), so this
   export contains 5,606 rows created after the 2021-01-01 extraction. Both analysis pipelines
   (`scripts/rerun-analysis.R`, `python/`) filter them out (keeping NULL `time_created`, which
   marks pre-column DC rows) — that is what makes the cleaned row count land exactly on the
   published 395,147. The `python/diagnose_post_cutoff.py` diagnostic shows those 5,606 rows
   are **not** echoes of the deployed estimator (median residual ~1.2 m from the published
   formula, matching genuine depth rows; ~1% within 5 cm), so they appear to be real depth
   estimates that continued to trickle in during 2021 — usable as bonus data in a refit, just
   not part of the 2021 dataset.
2. `tutorial` uses the 2021-faithful definition (the boolean flag; DC via
   `gsv_onboarding_pano` membership). Today's application additionally excludes labels by
   tutorial street edge, which the 2021 analysis did not.
3. The 2021 raw total (492,299) is not comparable to this export's raw total (468,608): the
   2021 CSVs included non-depth rows that the cleaning filters then removed. The comparable
   number is the post-cleaning count, which matches exactly.
4. Booleans are Postgres CSV `t`/`f`; both `readr` and the Python loader parse them.

## Regenerating

Run `scripts/extraction/extract-depth-labels.sh --dc-db sidewalk_dc` on the database host (see
script header), replace the `.csv.gz` files here, then re-run `scripts/rerun-analysis.R`
(refreshes `tests/fixtures/r-baseline/`), update the counts above, and run `pytest`.
