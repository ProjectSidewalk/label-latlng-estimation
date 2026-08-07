# The blend survives Mapillary: compression is gone, and most of the height residual is the rigs

**2026-08-07** · issue [#3](https://github.com/ProjectSidewalk/label-latlng-estimation/issues/3) (Stage 3, opening) · scores the candidate shipped by [Stages 1–2](2026-08-07-distance-refit.md) · feeds [SidewalkWebpage#4765](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4765) and [#4766](https://github.com/ProjectSidewalk/SidewalkWebpage/issues/4766)

| | |
|---|---|
| **−0.002 / +0.062** | the shipped blend's within-site range slope (m/m) on richmond / clovis — the compression signature #4766 measured at −0.59 for the deployed estimator is gone on the imagery it was never fit on |
| **−1.40** | the deployed raw-pixel model's range slope on clovis's uniform 2880-px panoramas — #4765's sign-flip, measured directly: at 2.3× below the calibration height the linear model is not degraded, it is broken |
| **−0.690 vs −0.24** | richmond height-residual slope, deployed model vs the three height-blind placements: only the deployed model reads pixels, and only it carries a pixel-frame height defect — 2.6× the band rig confounding alone produces |
| **69%** | of the held-out half's height slope removed by per-sequence camera heights fitted on a **disjoint** half of the sites (66–75% over five seeds; 91% in-sample). Most of the residual is transferable rig geometry — not all of it, and the in-sample figure flatters |
| **0.872 vs 1.022** | fitted relative distance scale, GoPro Max vs iSTAR Pulsar sequences: per-source camera height is real, measurable from the data, and ordered exactly as mount geometry predicts |

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
  defect or camera-height (rig) confounding? → **§7**: mostly confounding — the blend reads no
  pixels at all, it sits inside the band the other height-blind placements show, and fitted
  per-sequence camera heights remove 69% of it on **held-out** sites (not the 91% the
  in-sample fit reports).
- **Q6 — The Stage 4 generalization.** Can per-source camera height be measured from
  self-consistency alone, with no ground truth? → **§7**: yes, and it transfers — rigs
  separate exactly as mount geometry predicts (GoPro Max 13% below the car mast), scales
  fitted on one half of the sites flatten the other half, and clovis's one physical rig
  correctly yields nothing.

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

Two counts run through this report and they are not the same population. The fuse emits
**operational sites** (richmond 1,570, of which 8,098 member detections); the diagnostics need
at least two views to disagree, so they run on the **multi-member** subset — the two count
columns below, and every slope in §6–§7. §4's census percentages are over
all operational members (8,098 richmond / 8,626 clovis), because the census is about the
imagery, not the diagnostic.

| run | imagery | role | multi-member sites | site members (views) | panos seen | dominant capture |
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
  noise. (The `altitude → computed_altitude` shift has a ≈ +25–27 m median in both cities, but
  only clovis is tight about it — p10 23.7 / p90 26.5, against richmond's −5.7 / +38.8. Either
  way it is a vertical-datum offset with SfM drift on top, not information about camera height
  above ground, and nothing here uses it.)
- **The rig zoo is real but car-shaped.** Richmond's six pano heights (2048–6144) come from
  four rigs — a professional car mount (NCTECH iSTAR Pulsar, 11000×5500, 69% of site members),
  GoPro Max (21%), and two minor ones — across 8 creators. Clovis is a single creator driving
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
  `pano_height/6656`, both demeaned within site (#4765's diagnostic). Read it against the
  **height-blind band**: B, C and D take no `pano_height` input at all, so whatever slope
  *they* show is rig confounding refracted through each one's shape, never a pixel-frame
  error. Only A reads pixels, so only A can carry one. The band is not a single number —
  §7's table shows B, C and D disagreeing by up to 0.16 on the same run — so it bounds a
  region, not a floor;
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

## §7 · Second axis: most of the height residual is the rigs, and the rigs transfer

Within-site height slope, every run that has more than one pano height (clovis has one, so the
slope is undefined there and the summary records `null` rather than float noise):

| run | A status quo | B normalized | C cotangent | **D blend** |
|---|---:|---:|---:|---:|
| richmond (Mapillary) | **−0.690** | −0.269 | −0.228 | −0.236 |
| paterson (GSV) | −0.432 | +0.035 | +0.144 | +0.107 |
| gainesville (GSV) | −0.344 | +0.033 | +0.156 | +0.102 |
| bend (GSV) | −0.305 | +0.175 | +0.335 | +0.274 |
| sao_paulo (GSV) | −0.593 | −0.125 | +0.045 | +0.009 |

(SEs 0.011–0.087.) Read the last three columns as one band, not as a floor and a measurement.
**B, C and D take no `pano_height` input at all** — their predicted distance is a function of
the depression angle alone — so none of them *can* show a pixel-frame height defect, and the
spread between them (up to 0.16 on bend) is the same confound arriving through three different
distance curves. The load-bearing row is A, the only candidate that reads pixels: on richmond
it sits at −0.690, **2.6× outside** the height-blind band, and it is negative on every run
while the band is positive on three of four GSV controls. That is #4765's pixel-frame defect,
alive on Mapillary and pointing the way the mechanism predicts.

So D is not *certified* clean on this axis by sitting in the band — it is height-blind, it
could not be otherwise. The question the band leaves open is whether the band itself is really
camera height. That is what the rest of this section tests.

**Fitting the rigs.** One multiplicative distance scale per sequence (equivalently, one camera
height per outing — a Mapillary sequence is one rig on one drive), by alternating least squares
on the same multi-view objective, **relative scale only**. Only the 997 of 1,183 richmond sites
seen by ≥ 2 sequences enter the objective (7,186 of 7,711 members): a site seen by one sequence
scales with that sequence and so pulls its `k` toward zero with nothing opposing it — pure
degeneracy, no information about relative scale. The global axis stays the RampNet#101 trap and
stays anchored to the GSV fit.

| richmond rig | sequences | members | fitted k (median, rel. to run mean) |
|---|---:|---:|---:|
| NCTECH iSTAR Pulsar (car mount, 11000×5500) | 229 | 5,231 | 1.022 |
| GoPro Max (2048/2880 px) | 13 | 1,158 | **0.872** |
| unbranded (12288×6144) | 13 | 643 | 1.009 |

![Figure 18 — left: richmond within-site height slope per candidate, with the height-blind band shaded; the blend sits inside it, in-sample per-sequence heights collapse it, and the held-out bar (hatched) removes 69% of the held-out half's own slope. Right: fitted per-sequence distance scale by rig — the GoPro Max sequences separate cleanly below the car-mount rigs.](../figures/fig18-falsification-height-axis.png)

**In-sample, the collapse proves nothing on its own.** Applying the fitted scales takes the
blend's richmond height slope −0.236 → −0.022 and RMS/range 0.200 → 0.189. But pano height is
*constant within a sequence*, so one free parameter per sequence can absorb any height-correlated
systematic by construction — rig geometry and model error alike. A 91% collapse is what this fit
does to noise.

**Held out, most of it survives.** Fit the scales on a random half of the sites; score the
other half's 3,973 members, which the fit never saw:

| richmond, held-out half | RMS/range | range slope | height slope |
|---|---:|---:|---:|
| D blend, unscaled | 0.2013 | +0.003 ± 0.009 | −0.255 ± 0.021 |
| D blend, transferred k | 0.2012 | +0.018 ± 0.009 | **−0.080 ± 0.021** |

**69% of the held-out height slope is removed by scales fitted on disjoint sites** — 66–75%
across five seeds, all committed in `holdout_seed_sweep`. That is the finding: per-source camera
height is real, measurable without ground truth, and it *transfers*. Two honest riders come with
it. The residual −0.080 is still 3.8 SE from zero, so per-sequence height explains most but not
all of the height dependence. And RMS/range is flat out-of-sample (0.2013 → 0.2012) while it
improved 6% in-sample: the scales carry height-axis information specifically, not general
self-consistency — the in-sample RMS gain was the parameters.

Clovis is the control that makes this legible. Its two Mapillary model strings are one physical
camera, and the fit says so: k = 0.991 and 0.992, both within 1% of the run mean. With nothing
to measure, transferring its scales to held-out sites makes self-consistency slightly *worse*
(RMS/range 0.180 → 0.186) — exactly the null result a method that only fits noise would fail to
produce.

This is the empirical backing for the Stage 4 generalization the Stages 1–2 report proposed:
**a per-source (or per-sequence, where sequences are known) base camera height with the
per-type offsets kept.** On GSV nothing changes; on Mapillary the base height is identifiable
from exactly this kind of self-consistency, per city or per rig, without any ground truth —
with the caveat that it recovers most, not all, of the rig spread.

## §8 · What this settles, and what it deliberately does not

**Settled by this report:**

- The #4766 checklist item the depth data could not reach: the geometry-shaped refit
  **transfers across a 2.8× resolution extrapolation to imagery from different cameras, rigs,
  and an SfM pose pipeline** with no compression signature (§6, the falsifiable axis) and no
  pixel dependence to defect on (§7 — the blend reads only the depression angle).
- #4765's defect is re-confirmed on the deployed model at production scale, from
  self-consistency alone — the compression axis on both cities, and the height axis at 2.6×
  the height-blind band on richmond — and clovis shows its severity is not linear in the
  height ratio.
- **Per-source camera height is measurable without ground truth and transfers**: scales fitted
  on one half of richmond's sites remove 69% of the other half's height slope, order by rig as
  mount geometry predicts, and correctly find nothing on clovis's single physical rig. That is
  the empirical licence for the Stage 4 per-source-height generalization.
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
- **A clean bill of health for the blend on the height axis.** §7's band cannot deliver one:
  the blend takes no `pano_height` input, so its height slope is a consistency check, not a
  test it could have failed. The falsifiable axis for the blend is §6's, and the residual
  −0.080 the fitted rig heights leave behind is unexplained.

## Reproducing this report

```bash
pip install -r python/requirements.txt
python python/run_mapillary_falsification.py --write  # ~25 s, offline, deterministic
python python/falsification_figures.py                # figs 17-18
pytest tests/test_mapillary_falsification_findings.py # 26 findings, incl. re-derivation
```

No network anywhere: everything derives from the committed `data/falsification-*` inputs,
which `python/import_falsification_inputs.py` regenerates byte-identically from an
auto-labeler checkout at the recorded commit.

---

*Report generated with [Claude Code](https://claude.com/claude-code) (claude-fable-5); every
headline number is asserted by `tests/test_mapillary_falsification_findings.py` against
`data/falsification-summary.json`, which regenerates deterministically from the committed
data.*
