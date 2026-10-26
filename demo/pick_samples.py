#!/usr/bin/env python
"""Decide which scenes the demo offers, from cached verdicts (2026-10-13).

Earlier selection ranked scenes by the gap between mean scores, which was the
wrong test twice over: the verdict comes from whether a run of frames crosses
the threshold, not from the clip mean, and the threshold is per-domain
(0.15-0.66 measured) rather than the package's flat 0.5. Judging by means
against 0.5 made the model look far worse than it is.

This reads the cache written by build_cache.py -- real inference, calibrated
threshold -- and scores each scene on what an evaluator will actually see:

  correct = the pristine clip is called clean AND both manipulated clips are
            called manipulated

A scene where all three are right is demo-ready. A scene where the pristine
clip is flagged is the worst case to show, because the evaluator picks the
clip blind and a false alarm on the honest video is the one failure that
undermines the whole demo.

    python pick_samples.py                  # report + write demo_samples.json
    python pick_samples.py --top 6          # keep the best 6 scenes
"""
import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "demo_samples.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=None,
                    help="keep only the N best scenes (default: all correct ones)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    with open(os.path.join(HERE, "clips.json"), encoding="utf-8") as f:
        clips = {c["id"]: c for c in json.load(f)["clips"]}

    scenes = defaultdict(dict)
    missing = []
    for cid, c in clips.items():
        path = os.path.join(CACHE_DIR, f"{cid}.json")
        if not os.path.exists(path):
            missing.append(cid)
            continue
        with open(path, encoding="utf-8") as f:
            res = json.load(f)
        scenes[(c["dataset"], c["scene"])][c["variant"]] = {
            "id": cid, "verdict": res["verdict"], "truth": c["truth"],
            "peak": max(res["smoothed"]), "threshold": res["threshold"],
            "n_segments": len(res["segments"]),
            "correct": (res["verdict"] == c["truth"]),
        }
    if missing:
        print(f"{len(missing)} clips not cached yet -- run build_cache.py first")
        if not scenes:
            return

    rows = []
    for (dom, scene), v in scenes.items():
        if len(v) < 3:
            continue
        n_ok = sum(1 for x in v.values() if x["correct"])
        pristine_ok = v["pristine"]["correct"]
        # margin: how far the weakest manipulated clip clears the threshold,
        # relative to how close the pristine clip came to it
        man_peak = min(v[k]["peak"] for k in ("propainter", "minimax"))
        thr = v["pristine"]["threshold"]
        rows.append({"dataset": dom, "scene": scene, "n_correct": n_ok,
                     "pristine_ok": pristine_ok, "threshold": thr,
                     "pristine_peak": v["pristine"]["peak"],
                     "manipulated_min_peak": man_peak,
                     "margin": man_peak - v["pristine"]["peak"],
                     "clips": {k: v[k]["id"] for k in v}})

    rows.sort(key=lambda r: (r["n_correct"], r["pristine_ok"], r["margin"]), reverse=True)

    print(f"\n{'scene':46s}{'ok':>4}{'thr':>7}{'pris pk':>9}{'man pk':>8}{'margin':>8}")
    for r in rows:
        flag = "" if r["n_correct"] == 3 else ("  <- pristine flagged" if not r["pristine_ok"] else "  <- missed")
        print(f"{r['dataset'] + ' / ' + r['scene'][:32]:46s}{r['n_correct']}/3{r['threshold']:>7.3f}"
              f"{r['pristine_peak']:>9.3f}{r['manipulated_min_peak']:>8.3f}{r['margin']:>+8.3f}{flag}")

    good = [r for r in rows if r["n_correct"] == 3]
    keep = good[: args.top] if args.top else good
    print(f"\n{len(good)}/{len(rows)} scenes fully correct; keeping {len(keep)}")
    if not keep:
        print("WARNING: no scene is fully correct -- widen the candidate pool "
              "(select_demo_clips.py --per-domain 6) before building the demo")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"scenes": keep, "all": rows}, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
