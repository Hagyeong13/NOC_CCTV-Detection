#!/usr/bin/env python
"""Score every candidate demo clip with the real model, so the demo menu is
chosen from measurements rather than from hope (2026-10-12).

Runs the collaborator's standalone package end-to-end (video in, per-frame
score out) on each clip in clips.json, using that clip's own domain model and
the adapted checkpoint -- adapted, not zero-shot, because the paper's claim is
about adaptation and the demo should show the configuration it argues for.

What we are picking for: a scene is a good demo scene when the model separates
its pristine twin from its manipulated twins. A high score on the manipulated
clip means nothing on its own if the pristine clip scores just as high, and
that gap is exactly what an evaluator will probe. The summary therefore ranks
scenes by (lowest manipulated mean) - (pristine mean), not by raw score.

CPU only (~8 s/clip), so it does not contend with GPU work.

    python measure_clips.py                 # all clips in clips.json
    python measure_clips.py --zero-shot     # same, no-DA checkpoints
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z_SH = os.path.dirname(HERE)
PKG = os.path.join(os.path.dirname(Z_SH), "z-final-sh", "standalone_package")
MANIFEST = os.path.join(Z_SH, "eval_share", "new_dataset", "case_manifest_ext.csv")
OUT = os.path.join(HERE, "clip_measurements.csv")


def video_paths():
    """case -> absolute clip path, from the shared manifest."""
    out = {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["case"]] = os.path.join(Z_SH, r["video_path"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--clips", default=os.path.join(HERE, "clips.json"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    sys.path.insert(0, PKG)
    import run_inference as ri

    with open(args.clips, encoding="utf-8") as f:
        clips = json.load(f)["clips"]
    paths = video_paths()

    rows = []
    for i, c in enumerate(clips, 1):
        path = paths.get(c["id"])
        if not path or not os.path.exists(path):
            print(f"[{i}/{len(clips)}] {c['id']}: MISSING video -- skipped", flush=True)
            continue
        t0 = time.time()
        try:
            frames, smoothed, branch = ri.score_video(
                path, c["dataset"], ri.CKPT_ROOT_DEFAULT, use_adapted=not args.zero_shot)
        except Exception as e:                                   # noqa: BLE001
            print(f"[{i}/{len(clips)}] {c['id']}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        s = np.asarray(smoothed, dtype=float)
        rows.append({"id": c["id"], "dataset": c["dataset"], "scene": c["scene"],
                     "variant": c["variant"], "truth": c["truth"],
                     "n_frames": len(s), "mean": float(s.mean()),
                     "max": float(s.max()), "p90": float(np.percentile(s, 90)),
                     "secs": round(time.time() - t0, 1)})
        print(f"[{i}/{len(clips)}] {c['dataset']:12s} {c['scene'][:30]:30s} "
              f"{c['variant']:10s} mean={s.mean():.4f} max={s.max():.4f} "
              f"({rows[-1]['secs']}s)", flush=True)

    if not rows:
        sys.exit("no clip scored")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    by_scene = defaultdict(dict)
    for r in rows:
        by_scene[(r["dataset"], r["scene"])][r["variant"]] = r

    print("\n" + "=" * 78)
    print("scenes ranked by separation = min(manipulated mean) - pristine mean")
    ranked = []
    for key, v in by_scene.items():
        if "pristine" not in v:
            continue
        man = [v[k]["mean"] for k in ("propainter", "minimax") if k in v]
        if not man:
            continue
        ranked.append((min(man) - v["pristine"]["mean"], key, v))
    for gap, (dom, scene), v in sorted(ranked, reverse=True):
        parts = " ".join(f"{k[:4]}={v[k]['mean']:.3f}" for k in
                         ("pristine", "propainter", "minimax") if k in v)
        print(f"  gap={gap:+.4f}  {dom:12s} {scene[:30]:30s} {parts}")
    print(f"\nwrote {args.out}")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
