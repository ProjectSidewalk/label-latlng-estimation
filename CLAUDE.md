# CLAUDE.md — label-latlng-estimation

## Every artifact an experiment depends on gets archived — GitHub or Hugging Face, nothing else

Every input, intermediate, and output an experiment depends on must live either in this
repository (committed, with a `data/MANIFEST.md` entry) or in a Hugging Face dataset
pinned by revision and pointed to from the MANIFEST. Personal cloud storage (Drive,
Dropbox, share links) and "we can always re-fetch it" are not archival: links rot,
access changes hands, and remote services drift under you. This lab has had to rerun
whole experiments because an artifact everyone assumed would stay reachable didn't —
treat that as a law of nature, not bad luck.

Practical rules, all already in force in this repo:

- **Fetched bytes are committed verbatim** — the evidence itself, not a pointer to it —
  the way `data/depth-validation-tiles.jsonl.gz` and `data/triangulation-depth-payloads.jsonl.gz`
  do it. A re-fetch observes a different remote state; regenerate by intent only.
- **Network access is confined to explicit `fetch` stages.** `build`, figures, and tests
  must replay offline from a fresh checkout.
- **Bundles too large for GitHub** (roughly >50 MB) go to a Hugging Face dataset under
  the project's org, pinned by revision, with the pointer and regeneration instructions
  in `data/MANIFEST.md`.
- **A PR that adds an experiment is complete only when a fresh clone reproduces its
  numbers without asking anyone for files.** Findings tests lock the committed summaries
  so the artifacts and the claims cannot drift apart.

## A `label_id` is only a key next to its city

Project Sidewalk's database is **one schema per city**, and every schema's `label` table
numbers from 1. A `label_id` therefore identifies a label only alongside its `city`; on its
own it is a collision-rich integer, and the collisions are the common case rather than the
edge case. In the era corpus this repo ships, **317,596 of 468,608 rows (67.8%) carry a
`label_id` that also exists in another city**, one id is shared by all seven, and the column
holds 271,059 distinct values for 468,608 distinct labels. The 49-schema production census
behind `data/modern-truth-summary.json` is the same story at scale: 911,878 of 1,206,523
rows collide, up to 33 ways.

Nothing about the failure looks like a failure. `merge(..., on="label_id")` cross-joins and
*inflates* the frame instead of raising; `isin(set(other["label_id"]))` quietly returns True
for a row belonging to a different city. Both hand back a plausible frame of plausible
numbers, so it survives any test that samples rather than sweeps — a 40-row spot check
missed a 1.4% corruption here. This repo has shipped the bug twice: once in the modern-truth
joins (fixed in #15, `c2a7aa0`), and once in the depth pilot's `in_cleaned` flag, which was
wrong on 9.65% of era rows.

So, in this repo:

- **The key is `(city, label_id)`.** Materialize it as `label_uid = "city:label_id"` and
  assert `is_unique` in the same breath, the way `python/modern_truth.py` does — an
  unasserted composite key drifts back into a bare one during the next refactor.
- **Every label-keyed pandas join passes `validate=`** (`python/run_modern_truth.py:275`);
  the R pipeline joins `by = c('label_id', 'city')` (`scripts/rerun-analysis.R:105`).
- **Membership tests need the composite too.** Use
  `pd.MultiIndex.from_arrays([df["city"], df["label_id"]]).isin(...)`, never a `set()` of
  bare ids — this is the form the pilot bug took, and it reads as innocuous.
- **`pano_id` is Google's and IS globally unique**, as are `site_id` and Mapillary
  `sequence_id`. Joins on those are safe as written. The rule is specifically about
  per-schema database serials; presume it holds for any other one.
- **Committed artifacts and report prose carry the city with the id** — the label tables
  under `data/` do; in text write `cdmx:5701`, never a bare `5701`.
- **Uniqueness is asserted per city, deliberately** (`tests/test_data_contract.py:88`,
  `tests/test_depth_pilot_contract.py:131`). A global uniqueness assert on `label_id` would
  fail against correct data, which is exactly the point.

## Structure fitted inside one truth frame does not count until it is scored in another

A held-out split protects against overfitting the *rows*. It does nothing about overfitting
the *truth* — and every truth set in this project is a model output with its own systematics,
not a survey. Fit anything flexible enough and it will learn those systematics, score
beautifully on held-out rows drawn from the same pipeline, and evaporate the moment the
truth changes. This repo has been caught by exactly that twice:

- the era fit's **per-type camera-height table** was worth 4 cm on the era test split and
  **nothing at all** on modern truth (`reports/2026-08-07-modern-truth.md` §9); and
- the **entire 0.4 m GBM ceiling** of `reports/2026-08-07-gbm-ceiling.md` turned out to be
  the era truth's own resolution-conditioned scale. Once each side carried one modern
  parameter, the two-parameter closed form *beat* the booster
  (`reports/2026-08-10-gbm-transfer.md`).

So, before believing that a model has found real structure:

- **Ask what the truth's scale does along the axis the model leans on most.** Both failures
  above are visible in one line — `median(truth × tan depression)` cut by `pano_height` is
  2.80 m where the column is absent, 2.79 m at 6656 px and 2.35 m at 8192 px. A model given
  `pano_height` can answer on each subpopulation's own scale, which is worth a great deal
  inside that truth and nothing outside it. If a truth set's scale is *not* constant along
  that axis, assume the gain is calibration until shown otherwise.
- **Score in a second truth frame before the claim goes in a report headline** — modern
  truth (#3 Stage 4), Mapillary (#3 Stage 3), or bearing-only triangulation (#7). Where a
  second frame genuinely cannot host the test, say so in the report rather than letting the
  single-frame number stand unqualified.
- **Give both sides the same calibration budget.** Comparing a freshly-fitted model against
  a closed form carrying another era's constants measures the constants, not the models.
- **A per-group parameter needs a held-out split by group**, not by row: one free parameter
  per group absorbs any group-correlated systematic by construction. The falsification's
  per-sequence camera heights removed 91% of the residual in sample and 69% out of it.
