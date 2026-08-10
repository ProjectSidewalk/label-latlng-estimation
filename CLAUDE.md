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

## Structure fitted inside one truth frame does not count until it is scored in another

A held-out split protects against overfitting the *rows*. It does nothing about overfitting
the *truth* — and every truth set in this project is a model output with its own systematics,
not a survey. Fit anything flexible enough and it will learn those systematics, score
beautifully on held-out rows drawn from the same pipeline, and evaporate the moment the
truth changes. This repo has been caught by exactly that twice:

- the era fit's **per-type camera-height table** was worth 4 cm on the era test split and
  **nothing at all** on modern truth (`reports/2026-08-07-modern-truth.md` §7); and
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
