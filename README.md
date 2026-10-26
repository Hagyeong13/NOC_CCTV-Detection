# NOC_CCTV-Detection

Detecting object-removal inpainting forgery in CCTV footage with a five-branch
forensic fusion detector, plus the demo used for the graduation showcase.

Companion code for *A Multi-Cue Fusion Framework for Cross-Domain Inpainting
Forgery Detection in CCTV Videos* — Songhye Kim, Hagyeong Kim, Cheolsoo Park,
Jihyeon Ryu (Kwangwoon University).

## What the detector does

Object-removal inpainting can delete a person from surveillance footage and
rebuild the background convincingly enough that nothing looks wrong. The
forensic traces it leaves are not reliably visible as boundaries, so this
detector combines five kinds of evidence and fuses them per frame:

| Branch | Evidence | Model |
| --- | --- | --- |
| Statistical | 17 frame descriptors + 30 clip statistics (47-D) | LightGBM |
| Compression | 6 PRNU/block features + 30 clip statistics (36-D) | LightGBM |
| Residual | four forensic maps, 4×32×32 tensor | CNN + 1D-TCN |
| Local patch | full-resolution map crops, 4×16×16 per patch | shared CNN + MIL |
| Temporal frequency | per-frame spectra with activity input | MLP + MIL |

Each branch is calibrated against a **source-only CDF** before fusion, then the
five calibrated scores are averaged with equal weight and smoothed causally, so
a frame's score only ever uses past frames. Equal weights are deliberate:
learned fusion weights were tested and overfitted the small adaptation set.

Ablation puts the calibration step first by a wide margin — removing
Source-CDF costs 0.1411 macro AUC, far more than removing any single branch
(0.0300 at most).

## Benchmark and results

201 source scenes from LASIESTA, ShanghaiTech, CDnet2014, StreetScene and SBI,
each contributing one pristine sequence and two manipulated variants
(ProPainter and MiniMax-Remover) — 603 case variants, 130,625 labeled frames,
44.3% manipulated.

Leave-one-domain-out, six seeds, frame-level ROC-AUC:

- **0.7920 ± 0.0082** macro with target adaptation
- 0.6837 ± 0.0086 for the matched control without adaptation
- pooled 70/15/15: 0.8517 ± 0.0571 · in-domain: 0.8898 ± 0.0270

Splits are scene-group disjoint, so a scene's pristine and forged variants can
never straddle the train/adapt/test boundary.

**Read the LODO numbers as what they are.** The adapted model is given ~20% of
the held-out domain's scene groups as labeled adaptation data and is scored on
the remaining groups; the external baselines it is compared against use no
target labels. That is a system-level comparison under different label budgets,
not an architecture-only result, and the paper says so. Work matching the label
budget across methods is ongoing.

## What is in this repository

`model/` — inference code for the five-branch detector, written by the
co-authors and vendored unmodified (see `model/README_VENDOR.md`).

`demo/` — the showcase demo and the scripts that build it: per-domain threshold
calibration, clip selection by visual subtlety, score caching, a static
single-file HTML viewer, a Gradio app, and an occlusion-sensitivity probe.

## What is deliberately not here

**Trained checkpoints.** The five per-domain models are co-authored artifacts
and are not redistributed here. Request them from the authors.

**Video.** The demo clips are derived from the five source datasets, whose
terms do not permit redistribution. Get them from their own maintainers and
rebuild locally.

Without both, the demo scripts still run and will tell you what is missing
rather than inventing a score.

## Running it

Inference on one clip, given a checkpoint directory:

```bash
cd model
pip install -r requirements.txt
python3 run_inference.py --video /path/to/clip.mp4 --domain sbi
```

CPU only, roughly 8 seconds per clip; no GPU needed.

**The shipped per-domain models are in-domain models**, each trained on 70% of
its own domain and few-shot adapted on a further 15%. They are not the LODO
models the paper's cross-domain numbers come from. On a genuinely unseen
domain they fall to roughly 0.51 macro AUC — about chance. Treat them as tools
for "CCTV scenes resembling these five environments", which is also why the
demo offers a fixed clip menu instead of arbitrary upload.

The demo app and the static HTML build:

```bash
cd demo
python3 build_cache.py          # real inference, cached per clip
python3 build_html_demo.py      # -> html_demo/index.html, self-contained
python3 app.py                  # Gradio UI on 127.0.0.1:7860
```

`calibrate_thresholds.py` fits the per-domain decision thresholds. These are
not interchangeable with a flat 0.5 — calibrated at 5% FPR on the adaptation
split they range from 0.16 to 0.73 across the five domains, and the demo
reports per-frame scores against the calibrated value for that domain.

A frame crossing the threshold is not by itself a verdict: the demo requires a
run of at least 24 consecutive frames (1 s at 24 fps) before it reports an
interval, because a 10% frame-level false-positive budget puts about 20 frames
of a 200-frame clean clip over the line.

## Honesty notes

The demo shows the model's real output and nothing else. There is no
heatmap overlay: the Patch-MIL map localises only to a sixteenth of the frame,
which on screen read as a claim about *where* that the model cannot support —
and because its per-patch scores are a relative ranking, the busiest patch
scored highest even on untouched video, so the overlay tracked moving people
through pristine clips while the verdict correctly said "clean". What the model
does support is *when*, which the score curve and the interval table carry.
`occlusion_probe.py` is the honest alternative for *where*: it blanks one
region, re-runs the detector, and measures how far the score falls.
