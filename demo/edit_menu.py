#!/usr/bin/env python
"""Drop clips from the demo menu and re-number what is left (2026-10-15).

Two jobs:

  --drop takes scenes or individual clips out, for the ones that look wrong on
  screen even when the verdict is right.

  Re-numbering fixes a leak in the original labelling. "clip 1/2/3" was
  assigned in a fixed variant order, so clip 1 was the pristine one in every
  scene -- reveal a single scene and the evaluator knows the answer to all the
  others. Numbers are now shuffled per scene with a seed derived from the
  scene name, so the order is stable across rebuilds but carries no
  information.

    python edit_menu.py --drop train010 --drop Candela_m1.10_clip03_propainter
    python edit_menu.py --list
"""
import argparse
import hashlib
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
CLIPS = os.path.join(HERE, "clips.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", action="append", default=[],
                    help="scene name or clip id; repeatable")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    with open(CLIPS, encoding="utf-8") as f:
        payload = json.load(f)
    clips = payload["clips"]

    if args.list:
        for c in clips:
            print(f"  {c['label']:52s} {c['variant']:10s} {c['id']}")
        print(f"  {len(clips)} clips")
        return

    before = len(clips)
    kept = [c for c in clips
            if not any(d == c["id"] or d == c["scene"] for d in args.drop)]
    dropped = [c["id"] for c in clips if c not in kept]
    for cid in dropped:
        print(f"  dropped {cid}")

    by_scene = {}
    for c in kept:
        by_scene.setdefault((c["dataset"], c["scene"]), []).append(c)
    for (dom, scene), group in by_scene.items():
        # Sort by variant first: the permutation is applied positionally, so
        # without a fixed member order the mapping changes every rebuild (it
        # inherited the previous run's label sort). Scene name + variant set
        # must fully determine the numbering, or the answer key memorised
        # before a presentation stops matching the screen.
        group.sort(key=lambda c: c["variant"])
        seed = int(hashlib.sha1(scene.encode()).hexdigest()[:8], 16)
        order = list(range(len(group)))
        random.Random(seed).shuffle(order)
        for c, n in zip(group, order):
            c["label"] = f"{dom} / {scene[:28]} / clip {n + 1}"
        shown = ", ".join(f"clip {o + 1}={c['variant']}"
                          for c, o in zip(group, order))
        print(f"  {dom}/{scene}: {shown}")

    payload["clips"] = sorted(kept, key=lambda c: c["label"])
    with open(CLIPS, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n{before} -> {len(kept)} clips; wrote {CLIPS}")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
