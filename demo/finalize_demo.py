#!/usr/bin/env python
"""Cut the demo down to the scenes that work, then pack it (2026-10-13).

The candidate pool is deliberately wider than the demo: 37 scenes were scored
so that the menu could be chosen from measurements. This keeps only the scenes
where all three clips were called correctly -- pristine clean, both
manipulations flagged -- and drops the rest, because a menu entry that
misfires is worse than a shorter menu. An evaluator picks blind, so every
offered clip has to hold up.

Also balances across domains where possible: a menu that is all one domain
invites "does it only work on that camera?", which the five-domain spread
answers for free.

    python finalize_demo.py                  # keep all correct scenes
    python finalize_demo.py --per-domain 2   # at most 2 scenes per domain
    python finalize_demo.py --no-bundle      # skip the laptop bundle
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")
CLIPS = os.path.join(HERE, "clips.json")
SAMPLES = os.path.join(HERE, "demo_samples.json")
VARIANTS = ("pristine", "propainter", "minimax")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=2)
    ap.add_argument("--no-bundle", action="store_true")
    args = ap.parse_args()

    with open(SAMPLES, encoding="utf-8") as f:
        good = json.load(f)["scenes"]
    if not good:
        sys.exit("no scene passed -- nothing to finalize")

    by_dom = defaultdict(list)
    for s in good:                      # already sorted best-first
        by_dom[s["dataset"]].append(s)
    keep = [s for d in sorted(by_dom) for s in by_dom[d][: args.per_domain]]
    print(f"{len(good)} correct scenes -> keeping {len(keep)} "
          f"({', '.join(f'{d}:{min(len(v), args.per_domain)}' for d, v in sorted(by_dom.items()))})")

    with open(CLIPS, encoding="utf-8") as f:
        payload = json.load(f)
    keep_ids = {cid for s in keep for cid in s["clips"].values()}
    kept = [c for c in payload["clips"] if c["id"] in keep_ids]

    # renumber the neutral labels so they run 1..3 inside each kept scene and
    # still give nothing away about which variant is which
    order = {v: i for i, v in enumerate(VARIANTS)}
    for c in kept:
        c["label"] = f"{c['dataset']} / {c['scene'][:28]} / clip {order[c['variant']] + 1}"

    missing = [c["id"] for c in kept
               if not os.path.exists(os.path.join(CACHE_DIR, f"{c['id']}.json"))
               or not os.path.exists(os.path.join(CACHE_DIR, f"{c['id']}_overlay.mp4"))]
    if missing:
        sys.exit(f"cache incomplete for {len(missing)} clips: {missing[:4]} ... "
                 f"run build_cache.py first")

    shutil.copy2(CLIPS, os.path.join(HERE, "clips_all_candidates.json"))
    payload["clips"] = kept
    payload["note"] = ("menu restricted to scenes verified correct on all three "
                       "variants; full candidate pool in clips_all_candidates.json")
    with open(CLIPS, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"clips.json -> {len(kept)} clips ({len(keep)} scenes); "
          f"full pool saved to clips_all_candidates.json")

    # drop cache entries the menu no longer offers, so the bundle stays small
    dropped = 0
    for fn in os.listdir(CACHE_DIR):
        cid = fn.replace("_overlay.mp4", "").replace(".json", "")
        if cid not in keep_ids and fn != "index.json":
            os.remove(os.path.join(CACHE_DIR, fn))
            dropped += 1
    print(f"pruned {dropped} cache files for clips no longer offered")

    if args.no_bundle:
        print("ALL_DONE")
        return
    print("\nbuilding laptop bundle...")
    subprocess.run([sys.executable, os.path.join(HERE, "make_bundle.py"), "--zip"],
                   check=True)
    print("ALL_DONE")


if __name__ == "__main__":
    main()
