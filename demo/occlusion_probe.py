#!/usr/bin/env python
"""Occlusion sensitivity: where does the score actually come from? (2026-10-15)

The Patch-MIL map answers "which patch scored highest", which is a ranking
among patches and not the same question as "what made the clip look forged".
On screen it tracked whatever was moving. Occlusion asks the causal question
instead: blank out one region of every frame, re-run the whole detector, and
see how far the clip's score falls. A region whose removal costs the model its
evidence is a region the model was actually using.

This is a standard explainability probe, and it is honest in a way a prettier
map would not be -- the numbers come from the model's own behaviour under
intervention, not from the ground truth.

Cost is the catch: one full inference pass per cell. This script measures that
on a single clip first so the grid size can be chosen with the real number in
hand rather than a guess.

    python occlusion_probe.py --clip <id> --grid 4      # time it first
    python occlusion_probe.py --clip <id> --grid 8
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def occlude_video(src, dst, box, fill="blur"):
    """Copy src to dst with `box`=(x0,y0,x1,y1) blanked in every frame.
    Blur rather than a flat fill: a hard grey rectangle is itself a strong
    forensic anomaly and would confound the very signal being measured."""
    cap = cv2.VideoCapture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    wr = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    x0, y0, x1, y1 = box
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        roi = fr[y0:y1, x0:x1]
        if roi.size:
            k = max(3, (min(roi.shape[:2]) // 2) * 2 + 1)
            fr[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
        wr.write(fr)
    cap.release(); wr.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    import inference

    with open(os.path.join(HERE, "clips.json"), encoding="utf-8") as f:
        clips = {c["id"]: c for c in json.load(f)["clips"]}
    clip = clips[args.clip]
    src = inference.resolve_video(args.clip)
    dom = clip["dataset"]

    t0 = time.time()
    base = inference.analyze(src, dom)
    base_peak = float(np.max(base["smoothed"]))
    one_pass = time.time() - t0
    g = args.grid
    print(f"baseline peak score {base_peak:.4f}  ({one_pass:.0f}s per pass)")
    print(f"grid {g}x{g} = {g * g} cells -> about "
          f"{one_pass * g * g / 60:.0f} min for this clip", flush=True)

    cap = cv2.VideoCapture(src)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    drops = np.zeros((g, g), dtype=float)
    tmp = tempfile.mkdtemp(prefix="occl_")
    try:
        for gy in range(g):
            for gx in range(g):
                box = (gx * W // g, gy * H // g,
                       (gx + 1) * W // g, (gy + 1) * H // g)
                occ = os.path.join(tmp, f"occ_{gy}_{gx}.mp4")
                occlude_video(src, occ, box)
                r = inference.analyze(occ, dom)
                drops[gy, gx] = base_peak - float(np.max(r["smoothed"]))
                print(f"  cell ({gy},{gx}) peak {np.max(r['smoothed']):.4f} "
                      f"drop {drops[gy, gx]:+.4f}", flush=True)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print("\ndrop map (higher = the model relied on this region more):")
    print(np.round(drops, 4))
    out = args.out or os.path.join(HERE, f"occlusion_{args.clip}_{g}x{g}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"clip": args.clip, "grid": g, "baseline_peak": base_peak,
                   "drops": drops.tolist()}, f, indent=2)
    print(f"wrote {out}")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
