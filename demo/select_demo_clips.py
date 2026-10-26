#!/usr/bin/env python
"""Pick the clips the graduation demo offers, from held-out data only
(2026-10-09).

The demo must survive the obvious question -- "did it just memorise these?"
-- so every offered clip comes from the test portion of the group-safe
70/10/20 in-domain split, i.e. a scene the demo model never trains on. The
demo model is trained on the union of those splits' pretrain portions across
all five domains, so "test" here means held out for every domain at once.

Selection, per domain:
  - take scenes whose three variants (pristine / propainter / minimax) are
    ALL in the test portion, so the UI can offer the same scene both ways
    and let the evaluator pick blind
  - prefer scenes long enough for the pipeline: TempFreq needs a 32-frame
    FFT window and fusion smooths over 32 frames, so short clips are unusable
  - keep two scenes per domain

It also records which variant is which, so the UI can hide the answer until
after the model has spoken.

    python select_demo_clips.py                 # writes clips.json
    python select_demo_clips.py --per-domain 3
"""
import argparse
import json
import os
import sys
from collections import defaultdict

_DA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "domain_adaptation")
sys.path.insert(0, os.path.join(_DA, "src"))   # data, splits, paths
sys.path.insert(0, _DA)                        # the hybrid-row builder

import data
import splits
import paths
import run_experiment_extended_full_retrain as retrain

MIN_FRAMES = 80          # 32-frame FFT window + 32-frame smoothing + margin
VARIANTS = ("pristine", "propainter", "minimax")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clips.json")


def variant_of(case):
    low = case.lower()
    for v in VARIANTS:
        if low.endswith("_" + v):
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    orig_rows = data.load_rows()
    ext_rows_csv = data.load_rows(csv_path=paths.TRAIN_CSV_EXTENDED)
    casing = retrain.orig_casing_map(orig_rows)
    rows = retrain.build_hybrid_rows(orig_rows, ext_rows_csv,
                                    retrain.video_covered_cases(casing))
    print(f"hybrid rows={len(rows)} ({len({r['case'] for r in rows})} case-variants)")

    folds = splits.indomain_701020_splits(rows, seed=args.seed, verbose=False)

    picked = []
    for dom in sorted(folds):
        test_rows = folds[dom]["test"]
        # frames per case-variant, and which scene (group) it belongs to
        n_frames = defaultdict(int)
        group_of = {}
        for r in test_rows:
            n_frames[r["case"]] += 1
            group_of[r["case"]] = r["_group"]

        by_group = defaultdict(dict)
        for case, n in n_frames.items():
            v = variant_of(case)
            if v:
                by_group[group_of[case]][v] = {"case": case, "n_frames": n}

        # scenes where all three variants are held out and long enough
        complete = [(g, d) for g, d in by_group.items()
                    if len(d) == len(VARIANTS)
                    and min(x["n_frames"] for x in d.values()) >= MIN_FRAMES]
        complete.sort(key=lambda gd: -min(x["n_frames"] for x in gd[1].values()))

        if not complete:
            print(f"  {dom:14s} no scene has all three variants held out -- skipped")
            continue
        for g, d in complete[: args.per_domain]:
            picked.append({"dataset": dom, "scene": g,
                           "variants": {v: d[v]["case"] for v in VARIANTS},
                           "n_frames": min(x["n_frames"] for x in d.values())})
            print(f"  {dom:14s} {g:48s} {picked[-1]['n_frames']} frames")

    entries = []
    for p in picked:
        for v, case in p["variants"].items():
            entries.append({
                "id": case,
                "dataset": p["dataset"],
                "scene": p["scene"],
                "variant": v,
                "truth": "clean" if v == "pristine" else "manipulated",
                "n_frames": p["n_frames"],
                # the UI shows this instead of the case name, so the variant
                # is not given away before the model has answered
                "label": f"{p['dataset']} / {p['scene'][:28]} / clip {VARIANTS.index(v) + 1}",
            })

    payload = {"split": "indomain_70_10_20 test", "seed": args.seed,
               "min_frames": MIN_FRAMES, "scenes": len(picked), "clips": entries}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n{len(picked)} scenes -> {len(entries)} clips; wrote {args.out}")


if __name__ == "__main__":
    main()
