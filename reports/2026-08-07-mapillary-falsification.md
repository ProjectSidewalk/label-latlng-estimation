# The blend survives Mapillary: compression is gone, and the height residual is the rigs, not the pixels

**2026-08-07** · issue [#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3) (Stage 3, opening) · scores the candidate shipped by [Stages 1–2](2026-08-07-distance-refit.md) · feeds [SidewalkWebpage#4765](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4765) and [#4766](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4766)

| | |
|---|---|
| **−0.002 / +0.062** | the shipped blend's within-site range slope (m/m) on richmond / clovis — the compression signature #4766 measured at −0.59 for the deployed estimator is gone on the imagery it was never fit on |
| **−1.40** | the deployed raw-pixel model's range slope on clovis's uniform 2880-px panoramas — #4765's sign-flip, measured directly: at 2.3× below the calibration height the linear model is not degraded, it is broken |
| **−0.236 → −0.022** | richmond height-residual slope for the blend: already at the confound floor as shipped, and collapsing to zero once each sequence gets its own fitted camera height — the residual height dependence was the rigs, not the pixels |
| **0.871 vs 1.023** | fitted relative distance scale, GoPro Max vs iSTAR Pulsar sequences: per-source camera height is real, measurable from the data, and ordered exactly as mount geometry predicts |

> Reproduce every number here in one command each:
>
> ```bash
> python python/run_mapillary_falsification.py --write  # census + diagnostics + sequence heights
> python python/falsification_figures.py                # figures 17-18
> pytest tests/test_mapillary_falsification_findings.py # the findings, locked
> ```

## §1 · Goal

Stages 1–2 refit the estimator's distance half on 395k depth-derived GSV labels and shipped a
horizon-saturating cotangent (`D_blend`) with provisional coefficients. All of that evidence is
GSV at 6656/8192 px. Mapillary is where the deployed model is known to break (#4765 measured
the bias flipping sign there), where there is **no absolute ground truth**, and where the refit
is a 2.8× resolution extrapolation.

**The goal of this stage — what issue #3 specified — is to falsify, not fit:** subject the
shipped candidate, unmodified, to the two scale-free self-consistency diagnostics that
condemned the deployed model, on imagery the refit never saw, and either break it or certify
that it transfers. No coefficient in the shipped candidate was touched by anything in this
report.

## §2 · Questions

The report and its code (`python/mapillary_falsification.py`) were set up to answer six
questions, each locked by the findings tests:

- **Q1 — Preconditions.** Are the Mapillary panos true 2:1 equirects, which pose stream is
  trustworthy, and what rig/capture population is actually present? → **§4**: all equirect;
  SfM (`computed_*`) pose only; a car-shaped rig zoo with negligible on-foot capture.
- **Q2 — Harness validity.** Can this reimplementation of #4766's diagnostics (whose code was
  never committed) be trusted? → **§5**: yes — externally validated to within a few
  thousandths of #4766's published slopes, including its counterintuitive normalization
  result.
- **Q3 — The deployed model.** Does the compression signature #4766 measured reproduce on
  Mapillary from self-consistency alone, and does #4765's sign-flip mechanism appear?
  → **§6**: yes — −0.32 on richmond, and −1.40 on clovis's 2880-px panoramas.
- **Q4 — The blend, range axis.** Is the shipped blend flat where the deployed model
  compresses — does the refit transfer across the 2.8× resolution extrapolation? → **§6**:
  yes — |slope| ≤ 0.09 on all six runs, with one located, designed exception (sao_paulo's
  far-field tail).
- **Q5 — The blend, height axis.** Is the blend's residual height dependence a pixel-frame
  defect or camera-height (rig) confounding? → **§7**: confounding — at the confound floor as
  shipped, collapsing to ≈ 0 once each sequence gets a fitted camera height.
- **Q6 — The Stage 4 generalization.** Can per-source camera height be measured from
  self-consistency alone, with no ground truth? → **§7**: yes — rigs separate exactly as
  mount geometry predicts (GoPro Max 13% below the car mast).

Three things are out of scope by design — absolute scale on Mapillary, human-click behaviour,
and the near-horizon regime; **§8** says why, and where the evidence for each lives instead.

## §3 · Dataset

The inputs are the auto-labeler's fused multi-view curb-ramp sites for the two Mapillary-viewer
cities plus four GSV control runs, imported and committed as `data/falsification-*` at
auto-labeler commit `0bbd8e6` (all six runs fused with identical default parameters;
regeneration verified byte-identical on richmond; provenance in `data/MANIFEST.md`). Each
fused site is several independent detections of one physical curb ramp from different
panoramas — repeatable AI detections, not human clicks — with a GLS position, per-member
normalized pixel coordinates, and a ray bearing derived from the SfM pose.

| run | imagery | role | fused sites | site members (views) | panos seen | dominant capture |
|---|---|---|---:|---:|---:|---|
| richmond | Mapillary | test | 1,183 | 7,711 | 9,091 | iSTAR Pulsar car mount (11000×5500, 69% of members) over a four-rig zoo (heights 2048–6144) |
| clovis | Mapillary | test | 1,560 | 7,691 | 72,776 | one creator, one GoPro Fusion (5760×2880), two years |
| paterson | GSV | control | 6,995 | 25,229 | 34,427 | 8192-px GSV (6656 minority) |
| gainesville | GSV | control | 5,926 | 16,770 | 35,204 | 8192-px GSV (6656 minority) |
| bend | GSV | control | 11,655 | 47,180 | 78,560 | 8192-px GSV (6656 minority) |
| sao_paulo | GSV | control | 6,261 | 19,283 | 22,741 | 8192-px GSV (6656 minority) |

Three properties of this population, fixed before any scoring, bound what it can testify about:

1. **It is conditioned to the well-behaved regime.** The fuse gate drops members whose
   flat-ground range exceeds 25 m, so every view sits at depression ≥ ~5.9°. The near-horizon
   population that motivated the blend's linear tail is invisible here by construction; what is
   being tested is the model's shape where geometry is defensible.
2. **It is scale-free.** Site consensus is a mean of the same rays being scored, so shrinking
   every predicted distance buys residual for free (the
   [RampNet#101](https://github.com/ProjectSidewalk/RampNet/issues/101) trap). Every reported
   number is therefore a ratio or a within-site slope; absolute scale is deliberately not
   claimed, in either direction.
3. **It is curb ramps only** — the auto-labeler detects one label type, so the per-type height
   spread from Stage 1 is untestable here; `D_blend` runs with its CurbRamp height (2.783 m).

## §4 · The census: what the falsification is allowed to assume

Measured over every pano the runs saw (9,091 richmond / 72,776 clovis), before any diagnostic:

- **Projection is not a confound.** Every Mapillary pano in both cities is a true 2:1
  equirectangular (`spherical` / `equirectangular` camera types only) — the auto-labeler's
  `is_pano` enumeration filter held, so the pixel→angle conversion below is exact.
- **Only the SfM pose is usable.** Clovis's raw EXIF compass is literal zero on 56% of panos,
  and richmond's SfM (`computed_*`) moves raw GPS positions a median 2.9 m (p90 9.6 m). Every
  bearing and position in this report is the SfM one — scoring against raw EXIF would score GPS
  noise. (The `altitude → computed_altitude` shift is a consistent ≈ +25–27 m in both cities —
  a vertical-datum offset, not information about camera height above ground.)
- **The rig zoo is real but car-shaped.** Richmond's six pano heights (2048–6144) come from
  four rigs — a professional car mount (NCTECH iSTAR Pulsar, 11000×5500, 69% of site members),
  GoPro Max (21%), and two minor ones — under 8 creators. Clovis is a single creator driving
  one GoPro Fusion at 5760×2880 for two years (its two Mapillary model strings are the same
  physical camera; §7 confirms that from the data). **On-foot capture is negligible**: 17 of
  8,098 and 6 of 8,626 site members come from walking-speed sequences (classified from
  per-sequence gross speed and frame spacing; several rigs stamp `captured_at` at fixed
  intervals, so naive frame speed is untrustworthy — the census records both). The rig class
  that would break a car-height cotangent hardest barely exists in these cities, and
  per-sequence height only has to separate car-class mounts.
- **The GSV controls sit on the refit's own heights** — site members overwhelmingly on
  8192-px panos with a 6656-px minority — so any Mapillary-only failure would implicate the
  extrapolation, not the harness.

## §5 · The method, and why to believe this implementation

[#4766](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4766)'s two scale-free
diagnostics, reimplemented from its published description (its code was never committed). Each
candidate places every member from its own panorama — SfM position, the fused member's ray
bearing, and the candidate's predicted distance from the **exact depression angle**
`(y_normalized − 0.5) · 180°` — and is scored on how well views of the same site agree:

- **residuals** are each placement minus its site's mean placement (within-site demeaned by
  construction); slopes use the along-ray component;
- **range slope** (m/m): pooled OLS of along-ray residual on predicted range, both demeaned
  within site. Negative = far views under-shoot their peers = compression;
- **height slope**: pooled OLS of along-ray residual (per site mean range) on
  `pano_height/6656`, both demeaned within site. The **confound floor** is the
  height-normalized model B's slope — B has no pixel dependence, so its slope is what rig
  confounding alone produces (#4765's diagnostic);
- **RMS/range**: pooled 2D residual RMS over mean predicted range.

The four candidates: **A** the deployed raw-pixel linear model (production zoom-1
coefficients, canvas term at the canvas centre — reproduces #4765's worked table exactly),
**B** = A with #4765's height normalization, **C** a raw cotangent, **D** the shipped blend.
Two convention pins hold the whole thing to the ecosystem: C at the auto-labeler's 2.6 m
reproduces every stored member `range_m` to < 0.5 mm (asserted in the findings tests), and the
population, bearings, and pose are identical across candidates — differences are the distance
model alone, the same isolation Stage 1 used.

**External validation.** Where the metric cannot depend on the run's rig mix — B and C, whose
predictions ignore pano height, on the range axis — this implementation lands on #4766's
published numbers from a fuse of the same cities taken months earlier at half the corpus size:

| range slope (m/m) | #4766 published | this report |
|---|---|---|
| paterson · C cotangent | +0.0983 | **+0.0985** |
| paterson · B normalized | −0.4496 | **−0.4381** |
| richmond · C cotangent | +0.1207 | **+0.1188** |
| richmond · B normalized | −0.2901 | **−0.2931** |

That includes reproducing #4766's counterintuitive finding that the #4765 normalization alone
makes the GSV range axis *worse* (it holds on all four controls here). Model A's richmond
numbers moved (−0.59 → −0.32) — A is the one candidate whose behaviour is coupled to the rig
mix, and richmond's corpus doubled toward the 5500-px car rig since #4766's snapshot. The
height-slope column is stated under this report's definition only; #4766's height
normalization was not recorded, so those magnitudes are internally comparable here but not
across reports.

## §6 · First axis: the compression signature is gone

Within-site range slope, all six runs:

| run | views | A status quo | B normalized | C cotangent | **D blend** |
|---|---:|---:|---:|---:|---:|
| richmond (Mapillary) | 7,711 | −0.323 | −0.293 | +0.119 | **−0.002** |
| clovis (Mapillary) | 7,691 | **−1.403** | −0.164 | +0.158 | **+0.062** |
| paterson (GSV) | 25,229 | −0.235 | −0.438 | +0.099 | **−0.036** |
| gainesville (GSV) | 16,770 | −0.233 | −0.383 | +0.079 | **−0.053** |
| bend (GSV) | 47,180 | −0.105 | −0.208 | +0.099 | **−0.004** |
| sao_paulo (GSV) | 19,283 | −0.230 | −0.387 | +0.030 | **−0.090** |

(SEs 0.001–0.027; full matrix with RMS/range in `data/falsification-summary.json`.)

![Figure 17 — within-site range slope (m/m) for the four candidates on all six runs. The deployed model A sits below zero everywhere and falls off the chart on clovis; the shipped blend D clusters at zero on every run, including the two Mapillary cities it was never fit on.](../figures/fig17-falsification-range-axis.png)

Four readings:

- **The deployed model compresses everywhere, and clovis is the sign-flip measured.** At
  2880 px a raw-pixel offset subtends 2.3× the angle the coefficients assume, so the linear
  model's already-compressive shape is amplified into −1.40 m/m: within one site, a view 10 m
  further out under-places by 14 m relative to its peers. This is the mechanism behind
  #4765's +4.21 m clovis bias, seen without any external reference.
- **The blend is flat on the imagery it was never fit on.** |slope| ≤ 0.09 everywhere,
  −0.002 on the height-zoo city. It beats both linear models on every run and the raw
  cotangent on five of six.
- **The cotangent's consistent +0.03..+0.16 is the flat-ground residual** RampNet#101
  identified; the blend's near-zero slopes mean its shape absorbs most of that too.
- **The exception is honest and located**: on sao_paulo (−0.090 vs C's +0.030), whose member
  mix is the most far-heavy, the blend's linear tail (below 11.25°, ranges ≳ 14 m) under-shoots
  relative to the pure cotangent. That is the designed trade — boundedness at the horizon paid
  for in the far field — visible exactly where it should be and nowhere else.

## §7 · Second axis: the height residual is the rigs, and the rigs are measurable

On richmond (figure 18 below), the deployed model's
height slope (−0.690) sits 2.6× above the confound floor (B: −0.269) — the pixel-frame defect,
alive on Mapillary. The blend as shipped is **already at the floor** (−0.236): no measurable
pixel-frame height dependence beyond what a pixel-independent placement shows from rig
confounding alone.

Then the floor itself: one multiplicative distance scale per sequence (equivalently, one
camera height per outing — a Mapillary sequence is one rig on one drive), fitted by
alternating least squares on the same multi-view objective, **relative scale only** (997 of
1,183 richmond sites see ≥ 2 sequences, so relative rig scale is identified; the global axis
is the RampNet#101 trap and stays anchored to the GSV fit):

| richmond rig | sequences | members | fitted k (median, rel. to run mean) |
|---|---:|---:|---:|
| NCTECH iSTAR Pulsar (car mount, 11000×5500) | 231 | 5,312 | 1.023 |
| GoPro Max (2048/2880 px) | 13 | 1,489 | **0.871** |
| unbranded (12288×6144) | 13 | 760 | 1.010 |

![Figure 18 — left: richmond within-site height slope per candidate, with the confound floor marked; the blend sits at the floor as shipped and at ≈ 0 with per-sequence heights. Right: fitted per-sequence distance scale by rig — the GoPro Max sequences separate cleanly below the car-mount rigs.](../figures/fig18-falsification-height-axis.png)

With those per-sequence heights applied, the blend's richmond height slope collapses
−0.236 → **−0.022 ≈ 0** (range slope undisturbed at +0.005; RMS/range 0.200 → 0.188). So the
entire residual height dependence was **camera-height confounding** — pano height correlates
with rig, rig correlates with mount height — not a pixel-frame error in the model. The rigs
order exactly as mount geometry predicts (a GoPro on a low mount vs a telescoping car mast,
13% apart), and clovis cross-checks the method: its two model strings fit to k = 0.995 and
0.998 — one physical camera, correctly measured as one.

This is the empirical backing for the Stage 4 generalization the Stages 1–2 report proposed:
**a per-source (or per-sequence, where sequences are known) base camera height with the
per-type offsets kept.** On GSV nothing changes; on Mapillary the base height is identifiable
from exactly this kind of self-consistency, per city or per rig, without any ground truth.

## §8 · What this settles, and what it deliberately does not

**Settled by this report:**

- The #4766 checklist item the depth data could not reach: the geometry-shaped refit
  **transfers across a 2.8× resolution extrapolation to imagery from different cameras, rigs,
  and an SfM pose pipeline** with no compression signature and no pixel-frame height residual.
  Both scale-free diagnostics that condemned the deployed model now pass on its replacement.
- #4765's defect is re-confirmed on the deployed model at production scale, from
  self-consistency alone — and clovis shows its severity is not linear in the height ratio.
- The falsification harness itself is validated against #4766's independent implementation.

**Deliberately not claimed:**

- **Absolute accuracy on Mapillary.** Self-consistency provably cannot see a shared scale
  error. The absolute check is the modern-truth set: score the blend on post-2021 GSV labels
  against the panorama-tools#42 depth store when its backfill covers them — that is the
  remaining Stage 3 close-out item, and the census here (SfM-only pose, per-sequence heights)
  is the recipe for the Mapillary side of it.
- **Human-click behaviour.** These are repeatable AI detections; the crowd-noise question was
  settled separately by Stage 2's perturbation sweep.
- **The near-horizon regime** — excluded by the fuse gate (§3). Its evidence remains Stage 2's
  near-horizon table on GSV depth truth.

## Reproducing this report

```bash
pip install -r python/requirements.txt
python python/run_mapillary_falsification.py --write  # ~25 s, offline, deterministic
python python/falsification_figures.py                # figs 17-18
pytest tests/test_mapillary_falsification_findings.py # 20 findings, incl. re-derivation
```

No network anywhere: everything derives from the committed `data/falsification-*` inputs,
which `python/import_falsification_inputs.py` regenerates byte-identically from an
auto-labeler checkout at the recorded commit.

---

*Report generated with [Claude Code](https://claude.com/claude-code) (claude-fable-5); every
headline number is asserted by `tests/test_mapillary_falsification_findings.py` against
`data/falsification-summary.json`, which regenerates deterministically from the committed
data.*
