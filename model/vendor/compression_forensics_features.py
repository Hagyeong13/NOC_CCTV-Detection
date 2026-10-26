"""
5th expert branch (scoped down per user decision 2026-08-01): PRNU (camera
sensor noise pattern) correlation + local blockiness/quantization-residual --
NOT full H.264 bitstream parsing (motion vectors, skip-macroblock ratio),
which would need real bitstream access and is unlikely to carry signal given
this project's pristine/forged videos are each a SINGLE encoding generation
(no edit-recompress cycle for classical double-quantization detection to
exploit -- see project_codec_confound_finding.md memory for why).

Both features are computed purely on decoded pixels (blind, mask-free,
self-referenced -- same convention as blob_score/hp_prev_signed) and are
designed to be tested FIRST on the codec-controlled videos
(codec_controlled_datasets/) so they can't re-discover the encoder-identity
shortcut the codec-control investigation just controlled for.

1. PRNU correlation: estimate each video's own sensor-noise reference pattern
   as the per-pixel MEDIAN noise residual (frame - small-kernel-blur(frame))
   across all its frames (robust to a minority of forged frames/regions
   diluting the reference). Per frame, per 8x8-grid patch, compute normalized
   cross-correlation against the corresponding patch of that video's own
   reference. Hypothesis: generated/inpainted content doesn't reproduce the
   camera's sensor fingerprint, so the worst (lowest-correlation) patch in a
   forged frame should read as an anomaly relative to the same video's own
   baseline.

2. Blockiness: classic no-reference blocking-artifact index (Wang et al.) --
   mean abs pixel difference AT the 8-pixel block grid boundary minus the
   mean abs difference INSIDE blocks. Hypothesis: generated/inpainted texture
   quantizes differently than natural camera content even under an IDENTICAL
   global encoder setting (same QP/CRF), so local blockiness can differ from
   the rest of the same frame/video even with no codec-identity shortcut
   available.
"""
import sys

import cv2
import numpy as np

GRID = 8


def noise_residual(gray):
    return gray - cv2.GaussianBlur(gray, (3, 3), 0)


def prnu_reference_for_video(path, max_frames=None):
    """Per-pixel MEDIAN noise residual across the video's own frames -- a
    robust (minority-forged-frame-tolerant) estimate of that camera's sensor
    pattern, self-referenced, no external reference needed."""
    cap = cv2.VideoCapture(path)
    residuals = []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        residuals.append(noise_residual(gray))
        n += 1
        if max_frames and n >= max_frames:
            break
    cap.release()
    if not residuals:
        return None
    return np.median(np.stack(residuals, axis=0), axis=0)


def _patch_slices(h, w, grid=GRID):
    ph, pw = h // grid, w // grid
    slices = []
    for gy in range(grid):
        y0, y1 = gy * ph, (h if gy == grid - 1 else (gy + 1) * ph)
        for gx in range(grid):
            x0, x1 = gx * pw, (w if gx == grid - 1 else (gx + 1) * pw)
            slices.append((y0, y1, x0, x1))
    return slices


def _ncc(a, b, eps=1e-6):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + eps
    return float(np.dot(a, b) / denom)


def prnu_scores_for_video(path, reference, grid=GRID):
    """Returns (min_ncc_per_frame, mean_ncc_per_frame) -- min = worst
    (least camera-fingerprint-like) patch, the localized anomaly signal;
    mean = whole-frame baseline for self-reference normalization."""
    cap = cv2.VideoCapture(path)
    min_scores, mean_scores = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        res = noise_residual(gray)
        h, w = gray.shape
        patches = _patch_slices(h, w, grid)
        nccs = [_ncc(res[y0:y1, x0:x1], reference[y0:y1, x0:x1]) for (y0, y1, x0, x1) in patches]
        min_scores.append(min(nccs))
        mean_scores.append(float(np.mean(nccs)))
    cap.release()
    return min_scores, mean_scores


def _blockiness_1d(gray, axis, block=8):
    """Wang et al. no-reference blockiness: boundary discontinuity minus
    interior discontinuity, along one axis."""
    if axis == 1:
        gray = gray.T
    n = gray.shape[0]
    diffs = np.abs(np.diff(gray, axis=0))  # (n-1, w)
    is_boundary = (np.arange(1, n) % block == 0)
    if is_boundary.sum() == 0 or (~is_boundary).sum() == 0:
        return 0.0
    boundary_mean = diffs[is_boundary].mean()
    interior_mean = diffs[~is_boundary].mean()
    return float(boundary_mean - interior_mean)


def blockiness_for_frame(gray, grid=GRID):
    """Whole-frame blockiness index (higher = more visible 8x8 block edges)."""
    return 0.5 * (_blockiness_1d(gray, axis=0) + _blockiness_1d(gray, axis=1))


def blockiness_patch_max_for_frame(gray, grid=GRID):
    """Per-patch blockiness, returns the MAX across patches -- the most
    blocky-looking local region, a localized anomaly signal like prnu min."""
    h, w = gray.shape
    patches = _patch_slices(h, w, grid)
    vals = [blockiness_for_frame(gray[y0:y1, x0:x1]) for (y0, y1, x0, x1) in patches]
    return float(max(vals)), float(np.mean(vals))


def compression_scores_for_video(path, max_frames=None):
    """One-stop extraction: two passes (1st estimates the video's own PRNU
    reference, 2nd scores every frame against it + computes blockiness).
    Returns dict of per-frame lists: prnu_min, prnu_mean, block_max, block_mean."""
    reference = prnu_reference_for_video(path, max_frames)
    if reference is None:
        return {"prnu_min": [], "prnu_mean": [], "block_max": [], "block_mean": []}

    cap = cv2.VideoCapture(path)
    prnu_min, prnu_mean, block_max, block_mean = [], [], [], []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        res = noise_residual(gray)
        h, w = gray.shape
        patches = _patch_slices(h, w)
        nccs = [_ncc(res[y0:y1, x0:x1], reference[y0:y1, x0:x1]) for (y0, y1, x0, x1) in patches]
        prnu_min.append(min(nccs))
        prnu_mean.append(float(np.mean(nccs)))
        bmax, bmean = blockiness_patch_max_for_frame(gray)
        block_max.append(bmax)
        block_mean.append(bmean)
        n += 1
        if max_frames and n >= max_frames:
            break
    cap.release()
    return {"prnu_min": prnu_min, "prnu_mean": prnu_mean, "block_max": block_max, "block_mean": block_mean}


if __name__ == "__main__":
    path = sys.argv[1]
    scores = compression_scores_for_video(path, max_frames=30)
    for k, v in scores.items():
        arr = np.array(v)
        print(f"{k}: n={len(arr)} mean={arr.mean():.4f} std={arr.std():.4f} min={arr.min():.4f} max={arr.max():.4f}")
