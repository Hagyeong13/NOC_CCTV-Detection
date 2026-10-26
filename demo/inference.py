#!/usr/bin/env python
"""Demo-side wrapper around the collaborator's standalone inference package
(2026-10-12).

Two things the package does not expose on its own, both needed by the demo:

1. A spatial heatmap. Patch-MIL already scores each of the 4x4 patches of
   every frame; `predict_patch_mil` then keeps only the top-10% mean and
   throws the rest away. Rather than re-deriving those logits (which would
   risk drifting from what the model actually computed), we wrap that one
   function and keep what it computes on its way past. The displayed heatmap
   is therefore the model's own per-patch opinion, not a reconstruction.

   Resolution is 4x4 and nothing can change that -- it localises to roughly a
   sixteenth of the frame, not to a mask. The UI says so.

2. Suspicious intervals and a verdict. The project's segments.py does this,
   but importing it here would put two different `metrics` modules on the
   path at once -- the package vendors its own -- and the wrong one wins.
   The run-finding is a dozen lines, so it is reproduced below (same rule:
   maximal contiguous runs at or above the threshold, short runs dropped)
   rather than risking a silent cross-package import.

Everything else -- feature extraction, the five branches, source-CDF fusion,
causal smoothing -- is the package's code, called unmodified.
"""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z_SH = os.path.dirname(HERE)

# Two layouts: the dev tree (package two levels up) and the portable bundle
# shipped to the demo laptop (package sitting next to this file). Prefer the
# bundle so a copied folder works with no edits.
_BUNDLED_PKG = os.path.join(HERE, "standalone_package")
PKG = _BUNDLED_PKG if os.path.isdir(_BUNDLED_PKG) else os.path.join(
    os.path.dirname(Z_SH), "z-final-sh", "standalone_package")
VIDEO_DIR = os.path.join(HERE, "videos")        # bundle only

GRID = 4
FALLBACK_THRESHOLD = 0.5
MIN_RUN = 24         # 1 s at 24 fps: a suspicious interval must persist.
                     # An operating rule, not a tuned parameter -- a frame-level
                     # FPR budget on a 200-frame clip otherwise lets short bursts
                     # fire on honest video.
THRESHOLDS_JSON = os.path.join(HERE, "thresholds.json")


def domain_threshold(domain):
    """Per-domain cut-off calibrated at 10% FPR on that domain's adapt split
    (calibrate_thresholds.py). The package ships a flat 0.5, which is not what
    the method does and is badly wrong in both directions -- measured
    thresholds run from about 0.15 to 0.66 across the five domains, so one
    global value either misses almost everything or floods with false
    positives. Falls back to 0.5 only when the calibration has not been run."""
    try:
        with open(THRESHOLDS_JSON, encoding="utf-8") as f:
            d = json.load(f)["domains"]
        return float(d[domain]["threshold"]), True
    except Exception:                                        # noqa: BLE001
        return FALLBACK_THRESHOLD, False


def _ensure_paths():
    if PKG not in sys.path:
        sys.path.insert(0, PKG)


def find_segments(scores, frames, threshold, min_run=MIN_RUN):
    """Maximal contiguous runs at or above `threshold`, in frame order.
    Runs shorter than min_run are smoothing noise, not intervals."""
    scores = np.asarray(scores, dtype=float)
    flagged = scores >= threshold
    out, start = [], None
    for i in range(len(flagged) + 1):
        on = i < len(flagged) and bool(flagged[i])
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_run:
                run = scores[start:i]
                out.append({"start_frame": int(frames[start]),
                            "end_frame": int(frames[i - 1]),
                            "n_frames": int(i - start),
                            "mean_score": float(run.mean()),
                            "peak_score": float(run.max())})
            start = None
    return out


def _capture_calibrated():
    """fusion.fuse returns (smoothed, calibrated) but score_video discards the
    second. The calibrated values are the ones that actually enter the average
    -- raw branch scores are on wildly different scales (LGBM emits log-odds
    around -13..10, the neural branches sigmoid outputs in 0..1), so a
    "contribution" chart built from raw scores would be meaningless."""
    import fusion

    sink = {}
    original = fusion.fuse

    def patched(branch_scores_eval, reference, rows_eval, **kw):
        smoothed, calibrated = original(branch_scores_eval, reference, rows_eval, **kw)
        sink.update(calibrated)
        return smoothed, calibrated

    fusion.fuse = patched
    return (lambda: setattr(fusion, "fuse", original)), sink


def _capture_patch_logits():
    """Wrap pmm.predict_patch_mil so the per-patch logits it computes are kept.
    Returns (restore_fn, sink) where sink accumulates (batch, W*16) arrays."""
    import patch_mil_model as pmm
    import torch

    sink = []
    original = pmm.predict_patch_mil

    def patched(model, X_test, batch_size=64):
        model.eval()
        Xt = torch.from_numpy(X_test)
        scores = []
        with torch.no_grad():
            for i in range(0, Xt.shape[0], batch_size):
                xb = Xt[i:i + batch_size].to(pmm.DEVICE)
                b, w_, p_, c_, h_, ww_ = xb.shape
                flat = xb.view(b * w_ * p_, c_, h_, ww_)
                logits_flat = model(flat).view(b, w_ * p_)
                sink.append(logits_flat.cpu().numpy())
                bag_logit = pmm.topk_mean_logits(logits_flat, pmm.TOPK_FRAC)
                scores.append(torch.sigmoid(bag_logit).cpu().numpy())
        return np.concatenate(scores)

    pmm.predict_patch_mil = patched
    return (lambda: setattr(pmm, "predict_patch_mil", original)), sink


def _heatmaps_from_sink(sink, n_frames):
    """(n_frames, 4, 4) in [0,1]. make_windows uses stride=1 with half=W//2
    edge padding, so window i is centred on frame i+1 and the centre frame
    sits at position W//2 inside the window."""
    import patch_mil_model as pmm
    if not sink:
        return None
    logits = np.concatenate(sink, axis=0)                 # (N, W*16)
    w = pmm.W
    n_patches = logits.shape[1] // w
    logits = logits.reshape(-1, w, n_patches)[:, w // 2, :]
    if logits.shape[0] < n_frames:
        return None
    maps = 1.0 / (1.0 + np.exp(-logits[:n_frames]))
    return maps.reshape(n_frames, GRID, GRID)


def analyze(video_path, domain, adapted=True, threshold=None, work_dir=None):
    """Full demo result for one clip. Returns the dict the UI and the cache
    both consume."""
    _ensure_paths()
    import run_inference as ri

    thr_calibrated = False
    if threshold is None:
        threshold, thr_calibrated = domain_threshold(domain)

    keep = work_dir is not None
    work_dir = work_dir or tempfile.mkdtemp(prefix="demo_")
    restore_patch, sink = _capture_patch_logits()
    restore_cal, calibrated = _capture_calibrated()
    try:
        frames, smoothed, branch = ri.score_video(
            video_path, domain, ri.CKPT_ROOT_DEFAULT,
            use_adapted=adapted, work_dir=work_dir)
    finally:
        restore_patch()
        restore_cal()

    frames = [int(f) for f in frames]
    smoothed = np.asarray(smoothed, dtype=float)
    heat = _heatmaps_from_sink(sink, len(frames))

    segs = find_segments(smoothed, frames, threshold)

    # One 4x4 map for the whole clip, for the HTML demo: the per-frame maps are
    # too large to cache and, shown frame by frame, they track whatever is
    # moving. Averaging the baseline-corrected maps over the flagged frames
    # answers the question the display actually poses -- "where, overall, is
    # the suspicious region" -- while the score curve carries the "when".
    heat_agg = None
    if heat is not None:
        try:
            import overlay as _ov
            adj = _ov._prepare(heat, smoothed, threshold)
            flagged = smoothed >= threshold
            use = adj[flagged] if flagged.any() else adj
            heat_agg = [round(float(x), 5) for x in use.mean(axis=0).ravel()]
        except Exception:                                    # noqa: BLE001
            heat_agg = None

    # calibrated (post source-CDF, 0..1 and comparable); fall back to raw only
    # if the capture failed, in which case the chart is clearly off-scale
    branch_means = {}
    for k, v in (calibrated or branch or {}).items():
        arr = np.asarray(v, dtype=float)
        if arr.size:
            branch_means[k] = float(arr.mean())
    branch_means_calibrated = bool(calibrated)

    return {
        "video": os.path.basename(video_path),
        "domain": domain,
        "checkpoint": "adapted" if adapted else "zero-shot",
        "frames": frames,
        "smoothed": smoothed.tolist(),
        "threshold": float(threshold),
        "threshold_calibrated": thr_calibrated,
        "segments": segs,
        "verdict": "manipulated" if segs else "clean",
        "branch_means": branch_means,
        "branch_means_calibrated": branch_means_calibrated,
        "heatmap": heat.tolist() if heat is not None else None,
        "heatmap_agg": heat_agg,
        "work_dir": work_dir if keep else None,
    }


def resolve_video(clip_id):
    """Bundled copy if present, else the dev tree's manifest."""
    import csv
    local = os.path.join(VIDEO_DIR, f"{clip_id}.mp4")
    if os.path.exists(local):
        return local
    manifest = os.path.join(Z_SH, "eval_share", "new_dataset", "case_manifest_ext.csv")
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["case"] == clip_id:
                return os.path.join(Z_SH, r["video_path"])
    raise FileNotFoundError(f"no video for {clip_id}")


def run_clip(clip_id, bundle_dir=None):
    """Entry point app.py uses: resolve the clip's video and domain from
    clips.json plus the shared manifest, then analyze it."""
    import csv
    with open(os.path.join(HERE, "clips.json"), encoding="utf-8") as f:
        clips = {c["id"]: c for c in json.load(f)["clips"]}
    if clip_id not in clips:
        raise KeyError(f"unknown clip {clip_id}")
    return analyze(resolve_video(clip_id), clips[clip_id]["dataset"])
