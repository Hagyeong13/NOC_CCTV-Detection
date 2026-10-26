#!/usr/bin/env python
"""How visible is each manipulation to the naked eye? (2026-10-13)

The demo clips were chosen by how cleanly the model separates them, and that
criterion probably works against the demo: a manipulation the model finds easy
tends to be one that left strong artefacts, which is exactly the kind a viewer
spots unaided. "I can see it myself" is the worst reaction a detector demo can
get.

So measure it. Every scene has its pristine twin, so the manipulated region's
visibility is directly computable rather than a matter of opinion:

  PSNR  over the frames the manipulation touches -- higher means the forged
        frame is closer to the original, i.e. subtler
  SSIM  structural similarity on the same frames (implemented here; skimage
        is not in the environment and is not worth adding as a dependency)
  area  fraction of pixels that differ noticeably, which is what the eye
        actually catches -- a small well-blended patch reads as clean, a
        large smeared one does not

The goal is the intersection: subtle to look at AND still caught by the model.
If no clip is both, that is worth knowing too -- it would mean the demo can
only show cases a viewer could solve without us.

    python measure_visual_subtlety.py
"""
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z_SH = os.path.dirname(HERE)
LABELS = os.path.join(Z_SH, "eval_share", "new_dataset", "frame_labels_ext.csv")
POOL = os.path.join(HERE, "clips_all_candidates.json")
OUT = os.path.join(HERE, "visual_subtlety.csv")
DIFF_THRESH = 12          # 8-bit level difference a viewer plausibly notices
MAX_FRAMES = 60           # enough for a stable mean; keeps the sweep quick


def ssim(a, b):
    """Global SSIM on grayscale float images, standard constants."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2))
            / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def read_frames(path, idx):
    cap = cv2.VideoCapture(path)
    want, out = set(idx), {}
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = fr
        i += 1
        if len(out) == len(want):
            break
    cap.release()
    return out


def main():
    with open(POOL, encoding="utf-8") as f:
        clips = json.load(f)["clips"]
    import sys
    sys.path.insert(0, HERE)
    import inference

    pos = defaultdict(list)
    with open(LABELS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["gt_label"]) == 1:
                pos[r["case"]].append(int(r["frame"]))

    by_scene = defaultdict(dict)
    for c in clips:
        by_scene[(c["dataset"], c["scene"])][c["variant"]] = c

    rows = []
    for (dom, scene), v in sorted(by_scene.items()):
        if "pristine" not in v:
            continue
        try:
            p_path = inference.resolve_video(v["pristine"]["id"])
        except FileNotFoundError:
            continue
        for var in ("propainter", "minimax"):
            if var not in v:
                continue
            cid = v[var]["id"]
            frames = pos.get(cid, [])
            if not frames:
                continue
            step = max(1, len(frames) // MAX_FRAMES)
            # label frame f sits at 0-based video index f (row f <-> frame f+1,
            # and the first video frame carries no row)
            idx = [f for f in frames[::step]][:MAX_FRAMES]
            try:
                m_path = inference.resolve_video(cid)
            except FileNotFoundError:
                continue
            pf, mf = read_frames(p_path, idx), read_frames(m_path, idx)
            common = sorted(set(pf) & set(mf))
            if not common:
                continue
            psnrs, ssims, areas = [], [], []
            for i in common:
                a, b = pf[i], mf[i]
                if a.shape != b.shape:
                    b = cv2.resize(b, (a.shape[1], a.shape[0]))
                ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
                mse = float(np.mean((ga.astype(np.float64) - gb.astype(np.float64)) ** 2))
                psnrs.append(99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse))
                ssims.append(ssim(ga, gb))
                areas.append(float(np.mean(np.abs(ga.astype(int) - gb.astype(int))
                                           > DIFF_THRESH)))
            rows.append({"dataset": dom, "scene": scene, "variant": var,
                         "psnr": round(float(np.mean(psnrs)), 2),
                         "ssim": round(float(np.mean(ssims)), 4),
                         "diff_area": round(float(np.mean(areas)), 4),
                         "n_frames": len(common)})
            print(f"{dom:12s} {scene[:26]:26s} {var:10s} "
                  f"PSNR {rows[-1]['psnr']:5.1f}  SSIM {rows[-1]['ssim']:.3f}  "
                  f"diff {100 * rows[-1]['diff_area']:5.1f}%", flush=True)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {OUT}")
    print("Higher PSNR / SSIM and lower diff-area = harder to see unaided.")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
