#!/usr/bin/env python
"""Occlusion sensitivity over every demo clip, with controls (2026-11-02).

Running the probe only on manipulated clips would prove nothing: a score that
falls when a region is blurred might be falling because that region carried the
forgery, or simply because blurring anything perturbs the model. So this runs
the pristine sibling of every scene as a control -- if clean clips show drops
of the same size and concentration, the maps mean nothing.

Where a ground-truth mask exists (cdnet, sbi, streetscene; LASIESTA uses
clip-level labels because its masks are unreliable), the manipulated area is
projected onto the same 4x4 grid, so "the map points at the forged region" can
be checked instead of asserted.

    python run_occlusion_all.py --grid 4
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z_SH = os.path.dirname(HERE)
ZSH_ALT = "/data/workspace/guest-bcml/z.sh"
MANIFEST = os.path.join(Z_SH, "eval_share", "new_dataset", "case_manifest_ext.csv")


def mask_cell_weights(mask_path, grid):
    """Fraction of manipulated pixels falling in each grid cell, summed over
    the mask clip. The manifest stores masks as an mp4, not a folder of PNGs,
    so this reads frames rather than files. Returns None when absent."""
    import cv2
    if not mask_path or not os.path.isfile(mask_path):
        return None
    cap = cv2.VideoCapture(mask_path)
    if not cap.isOpened():
        return None
    acc = np.zeros((grid, grid), dtype=float)
    used = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        m = fr if fr.ndim == 2 else cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        pos = (m > 127).astype(float)
        if pos.sum() == 0:
            continue
        h, w = m.shape
        for gy in range(grid):
            for gx in range(grid):
                acc[gy, gx] += pos[gy * h // grid:(gy + 1) * h // grid,
                                   gx * w // grid:(gx + 1) * w // grid].sum()
        used += 1
    cap.release()
    if used == 0 or acc.sum() == 0:
        return None
    return acc / acc.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--only", help="쉼표로 구분한 clip id (기본: 전체)")
    args = ap.parse_args()

    with open(os.path.join(HERE, "clips.json"), encoding="utf-8") as f:
        clips = json.load(f)["clips"]
    if args.only:
        want = set(args.only.split(","))
        clips = [c for c in clips if c["id"] in want]

    man = {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            man[(r["dataset"], r["case"])] = r

    results = {}
    t_all = time.time()
    for i, c in enumerate(clips, 1):
        cid = c["id"]
        out = os.path.join(HERE, f"occlusion_{cid}_{args.grid}x{args.grid}.json")
        if os.path.exists(out):
            print(f"[{i}/{len(clips)}] {cid}: 기존 결과 사용", flush=True)
        else:
            print(f"[{i}/{len(clips)}] {cid}: 실행 중...", flush=True)
            r = subprocess.run([sys.executable,
                                os.path.join(HERE, "occlusion_probe.py"),
                                "--clip", cid, "--grid", str(args.grid),
                                "--out", out], capture_output=True, text=True)
            if r.returncode != 0:
                print(f"    실패: {r.stderr.strip().splitlines()[-1:]}", flush=True)
                continue
        with open(out, encoding="utf-8") as f:
            j = json.load(f)
        drops = np.array(j["drops"], dtype=float)

        mp = man.get((c["dataset"], cid), {}).get("mask_path") or ""
        weights = None
        for root in (Z_SH, ZSH_ALT):
            weights = mask_cell_weights(os.path.join(root, mp), args.grid) if mp else None
            if weights is not None:
                break
        results[cid] = dict(variant=c["variant"], dataset=c["dataset"],
                            baseline_peak=j["baseline_peak"],
                            drops=drops.tolist(),
                            mask_weights=None if weights is None else weights.tolist())
        print(f"    peak={j['baseline_peak']:.4f}  max drop={drops.max():+.4f} "
              f"at {np.unravel_index(drops.argmax(), drops.shape)}"
              f"  마스크={'있음' if weights is not None else '없음'}", flush=True)

    with open(os.path.join(HERE, f"occlusion_summary_{args.grid}x{args.grid}.json"),
              "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n총 {time.time() - t_all:.0f}s")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
