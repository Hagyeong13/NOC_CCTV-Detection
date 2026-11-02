#!/usr/bin/env python
"""Per-domain decision thresholds at 10% FPR (2026-10-13).

The standalone package ships a flat 0.5 cut-off, which is not what the method
does: this project calibrates per deployment domain on that domain's own
labelled adapt set, at TARGET_FPR = 0.10. The difference is not cosmetic. On
the demo candidates, lasiesta separates cleanly (pristine 0.066 vs
manipulated 0.216 / 0.370) yet every clip lands under 0.5, so a flat cut-off
reports "clean" for all three and the demo looks broken when the model is
actually fine.

Calibrating on the adapt split is the honest choice: it is the few-shot
labelled data the method already assumes, and it is disjoint from the test
split the demo clips come from.

Caveat worth keeping in view: streetscene's adapt split is only 3 clips, so
its threshold rests on very little. The report prints each domain's clip and
frame counts so a thin one is visible rather than silently trusted.

    python calibrate_thresholds.py            # -> thresholds.json
    python calibrate_thresholds.py --fpr 0.05
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z_SH = os.path.dirname(HERE)
DA = os.path.join(Z_SH, "domain_adaptation")
LABELS = os.path.join(Z_SH, "eval_share", "new_dataset", "frame_labels_ext.csv")
OUT = os.path.join(HERE, "thresholds.json")


def load_labels():
    lab = {}
    with open(LABELS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab[(r["dataset"], r["case"], int(r["frame"]))] = int(r["gt_label"])
    return lab


_ENUMERATE = """
import json, sys
sys.path.insert(0, {src!r}); sys.path.insert(0, {da!r})
import data, splits, paths
import run_experiment_extended_full_retrain as retrain
orig = data.load_rows()
ext = data.load_rows(csv_path=paths.TRAIN_CSV_EXTENDED)
rows = retrain.build_hybrid_rows(
    orig, ext, retrain.video_covered_cases(retrain.orig_casing_map(orig)))
folds = splits.indomain_701020_splits(rows, seed={seed}, verbose=False)
print("JSON:" + json.dumps(
    {{d: sorted({{r["case"] for r in f["adapt"]}}) for d, f in folds.items()}}))
"""


def adapt_clips(seed=0):
    """Which clips are in each domain's adapt split.

    Deliberately run in a child process. The split logic lives in
    domain_adaptation, which has its own data.py / paths.py / tempfreq_branch.py,
    and the inference package vendors modules with the same names. Importing
    both into one interpreter lets the first one win -- the failure mode is a
    TypeError deep inside scoring, not an import error, so it is worth the
    subprocess to make the collision impossible."""
    import subprocess
    code = _ENUMERATE.format(src=os.path.join(DA, "src"), da=DA, seed=seed)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout
    for line in out.splitlines():
        if line.startswith("JSON:"):
            return json.loads(line[5:])
    raise RuntimeError(f"could not enumerate adapt clips:\n{out[-500:]}")


def threshold_at_fpr(y, s, target_fpr):
    """Largest operating point with FPR at or below target. Mirrors the
    project's recall_at_fpr / calibrate_threshold convention: searchsorted
    with side='right' minus one, no interpolation."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    i = int(np.searchsorted(fpr, target_fpr, side="right") - 1)
    i = max(0, min(i, len(thr) - 1))
    return float(thr[i]), float(fpr[i]), float(tpr[i])


def all_criteria(y, s):
    """Every threshold rule on one ROC, so the choice can be argued rather
    than assumed. Textbooks usually teach Youden's J or the point closest to
    (0,1); this project fixes the false-positive rate instead, because in a
    forensic setting a false alarm and a miss do not cost the same and the
    alarm rate is the operational constraint. Reporting all of them makes the
    cost of that choice visible instead of hidden."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    out = {}

    for target in (0.05, 0.10):
        i = int(np.searchsorted(fpr, target, side="right") - 1)
        i = max(0, min(i, len(thr) - 1))
        out[f"fixed_fpr_{target:g}"] = (float(thr[i]), float(fpr[i]), float(tpr[i]))

    j = int(np.argmax(tpr - fpr))
    out["youden_j"] = (float(thr[j]), float(fpr[j]), float(tpr[j]))

    d = int(np.argmin(fpr ** 2 + (1 - tpr) ** 2))
    out["closest_to_01"] = (float(thr[d]), float(fpr[d]), float(tpr[d]))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--dump-scores",
                    help="adapt-split 프레임 점수를 이 JSON에 저장 (재계산용)")
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    import inference

    lab = load_labels()
    clips = adapt_clips(args.seed)
    dumps = {}
    result = {"target_fpr": args.fpr, "split": f"indomain_70_10_20 adapt (seed {args.seed})",
              "domains": {}}

    for dom in sorted(clips):
        ys, ss, used = [], [], 0
        t0 = time.time()
        for case in clips[dom]:
            try:
                path = inference.resolve_video(case)
            except FileNotFoundError:
                continue
            try:
                res = inference.analyze(path, dom)
            except Exception as e:                            # noqa: BLE001
                print(f"  {case}: FAILED {type(e).__name__}", flush=True)
                continue
            used += 1
            for fr, sc in zip(res["frames"], res["smoothed"]):
                key = (dom, case, int(fr))
                if key in lab:
                    ys.append(lab[key]); ss.append(float(sc))
        if not ys or len(set(ys)) < 2:
            print(f"{dom:14s} SKIPPED (clips={used}, frames={len(ys)}, "
                  f"classes={len(set(ys))})", flush=True)
            continue
        crit = all_criteria(np.array(ys), np.array(ss))
        result["domains_criteria"] = result.get("domains_criteria", {})
        result["domains_criteria"][dom] = {
            k: {"threshold": v[0], "fpr": v[1], "recall": v[2]}
            for k, v in crit.items()}
        # Keep the raw adapt-split scores: re-deriving a threshold rule should
        # not require another 20 minutes of inference.
        if args.dump_scores:
            dumps[dom] = {"y": ys, "s": ss}
        print(f"  [{dom}] 기준별 임계값")
        for k, (t_, f_, r_) in crit.items():
            print(f"     {k:16s} thr={t_:.4f}  FPR={f_:.4f}  recall={r_:.4f}")
        thr, fpr, tpr = threshold_at_fpr(np.array(ys), np.array(ss), args.fpr)
        result["domains"][dom] = {"threshold": thr, "achieved_fpr": fpr,
                                  "recall_at_threshold": tpr, "n_clips": used,
                                  "n_frames": len(ys), "pos_rate": float(np.mean(ys))}
        print(f"{dom:14s} thr={thr:.4f}  FPR={fpr:.3f}  recall={tpr:.3f}  "
              f"clips={used:3d} frames={len(ys):6d} pos={np.mean(ys):.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if args.dump_scores and dumps:
        with open(args.dump_scores, "w", encoding="utf-8") as f:
            json.dump(dumps, f)
        print(f"wrote {args.dump_scores} (adapt-split 점수)")
    print(f"\nwrote {args.out}")
    thin = [d for d, v in result["domains"].items() if v["n_clips"] < 6]
    if thin:
        print(f"NOTE: thin calibration sets (<6 clips): {', '.join(thin)} "
              f"-- treat those thresholds as provisional")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
