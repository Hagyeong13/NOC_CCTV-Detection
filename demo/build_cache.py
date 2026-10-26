#!/usr/bin/env python
"""Precompute every demo clip's result and overlay (2026-10-12).

Run this before the showcase. Each clip costs ~15 s of CPU here and more on a
laptop, and a progress bar that stalls mid-presentation is the one failure
mode worth spending an evening to remove. The cached artefacts are the output
of the real pipeline, not stand-ins, so nothing about the demo becomes fake by
running this -- it just moves the computation to a time when nobody is
watching.

    python build_cache.py                 # every clip, skipping cached ones
    python build_cache.py --force         # recompute all
    python build_cache.py --only cdnet    # one domain
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None, help="limit to one domain")
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    import app as demo_app

    clips = [c for c in demo_app.CLIPS.values()
             if args.only is None or c["dataset"] == args.only]
    os.makedirs(CACHE_DIR, exist_ok=True)

    done = skipped = failed = 0
    t_start = time.time()
    for i, c in enumerate(clips, 1):
        cid = c["id"]
        res_p = os.path.join(CACHE_DIR, f"{cid}.json")
        vid_p = os.path.join(CACHE_DIR, f"{cid}_overlay.mp4")
        if args.force:
            for p in (res_p, vid_p):
                if os.path.exists(p):
                    os.remove(p)
        elif os.path.exists(res_p) and os.path.exists(vid_p):
            skipped += 1
            continue

        t0 = time.time()
        res, vid, err = demo_app.analyze(cid)
        if err:
            print(f"[{i}/{len(clips)}] {cid}: FAILED -- {err.splitlines()[0]}", flush=True)
            failed += 1
            continue
        done += 1
        print(f"[{i}/{len(clips)}] {c['dataset']:12s} {c['variant']:10s} "
              f"{res['verdict']:12s} segs={len(res['segments'])} "
              f"mean={sum(res['smoothed'])/len(res['smoothed']):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\ncomputed {done}, skipped {skipped}, failed {failed} "
          f"in {(time.time()-t_start)/60:.1f} min")
    with open(os.path.join(CACHE_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"clips": sorted(demo_app.CLIPS), "built": time.strftime("%F %T")},
                  f, ensure_ascii=False, indent=2)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
