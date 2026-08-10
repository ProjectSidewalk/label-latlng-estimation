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
