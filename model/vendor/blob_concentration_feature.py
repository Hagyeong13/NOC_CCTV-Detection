"""
Prototype a new localized feature: for each frame, divide it into a grid of
patches, compute each patch's mean abs frame-to-frame difference, then take
blob_score = max_patch_diff / (median_patch_diff + eps).

Idea: a well-executed inpainting removal (static OR dynamic object) creates a
region that behaves DIFFERENTLY from the rest of the frame -- either much
smoother (static-object case, scene01) or much choppier/blurrier (dynamic-
object case). Either way the CHANGE should be spatially CONCENTRATED into one
blob rather than spread uniformly like camera noise or global lighting change.
Using a ratio (max/median) instead of the raw difference magnitude makes this
invariant to whether the video is generally high or low motion -- directly
targeting the confound found in scene01.
"""
import os
import sys
import glob
import cv2
import numpy as np

GRID = 8  # 8x8 patch grid


def _blob_score(gray, prev_gray):
    diff = np.abs(gray - prev_gray)
    h, w = diff.shape
    ph, pw = h // GRID, w // GRID
    cropped = diff[:ph*GRID, :pw*GRID]
    patch_means = cropped.reshape(GRID, ph, GRID, pw).mean(axis=(1, 3)).ravel()
    return patch_means.max() / (np.median(patch_means) + 1e-3)


def blob_scores_for_video(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    prev_gray = None
    scores = []
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None:
            scores.append(_blob_score(gray, prev_gray))
        prev_gray = gray
        n += 1
        if max_frames and n >= max_frames:
            break
    cap.release()
    return scores


def blob_scores_for_frame_dir(frame_dir, max_frames=None):
    paths = sorted(glob.glob(os.path.join(frame_dir, "*.png"))) or sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    if max_frames:
        paths = paths[:max_frames]
    prev_gray = None
    scores = []
    for p in paths:
        frame = cv2.imread(p)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None:
            scores.append(_blob_score(gray, prev_gray))
        prev_gray = gray
    return scores


if __name__ == "__main__":
    path = sys.argv[1]
    scores = blob_scores_for_video(path)
    print(f"n={len(scores)} mean={np.mean(scores):.3f} median={np.median(scores):.3f}")
