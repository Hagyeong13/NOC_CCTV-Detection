"""
BiSFD experiment: pool LASIESTA + ShanghaiTech + CDnet(train+heldout) into one
table, split 80:20 at scene/window (leakage-safe) level -- stratified per
dataset -- and train/evaluate with three new self-referenced feature families
on top of the existing base+context+signed feature set:

  A. bidirectional over-consistency  (<feat>_overstable)
  B. cross-feature relation consistency (rel_<a>_<b>)
  C. temporal event trajectory (persistence_run_length, onset_strength,
     recovery_strength, accel)

Reuses cw.BASE_FEATURES / cw.CONTEXT_FEATURES / cw.case_context and the same
LGBMRanker + per-video self-reference approach as retrain_eval_honest_cdnet_v4.py.

NOTE: this pooled 80:20 split is a different protocol from the CDnet
category-held-out / LASIESTA LOOCV numbers elsewhere in this project -- not
directly comparable. The "base_context" ablation stage on the SAME split is
the fair baseline to compare the new feature families against.
"""
import hashlib
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, "/home/tako/songhye/project4")

import numpy as np
from sklearn.metrics import roc_auc_score
import compare_weight_models as cw
import lightgbm as lgb

BASE = "/home/tako/songhye/project4/outputs_combined_lasiesta_static/feature_context_report/blob_features"
SEED = 0
ACT_THRESH = 2.7  # same activity threshold as retrain_eval_honest_cdnet_v4.py
PARAMS = dict(n_estimators=1500, num_leaves=15, learning_rate=0.03)
ROLL_WIN = 21

OVERSTABLE_FEATURES = ["rgb_prev", "stable", "transition", "texture_loss"]
RELATION_PAIRS = [
    ("rgb_prev", "transition"),
    ("hp_prev", "texture_loss"),
    ("transition", "stable"),
    ("texture_loss", "stable"),
    ("edge_prev", "transition"),
    ("dct_low", "hp_prev"),
]

NEW_FEATURE_GROUPS = {
    "overconsistency": [f"{feat}_overstable" for feat in OVERSTABLE_FEATURES],
    "relation": [f"rel_{a}_{b}" for a, b in RELATION_PAIRS],
    "temporal": ["persistence_run_length", "onset_strength", "recovery_strength", "accel"],
}


# =========================================================
# 1. Load + pool
# =========================================================
def load_pool():
    lasiesta = [
        r for r in cw.read_csv(f"{BASE}/frame_feature_training_table_with_boundary.csv")
        if r["case"].startswith(("LAS_", "O_", "PRISTINE_O_", "PRISTINE_LAS_"))
    ]
    cdnet_train = cw.read_csv(f"{BASE}/cdnet_train_v2_rows_with_boundary.csv")
    cdnet_heldout = cw.read_csv(f"{BASE}/cdnet_heldout_v2_rows_with_boundary.csv")
    shanghaitech = cw.read_csv(f"{BASE}/shanghaitech_rows_with_boundary.csv")

    for r in lasiesta:
        r["dataset"] = "lasiesta"
    for r in cdnet_train:
        r["dataset"] = "cdnet"
    for r in cdnet_heldout:
        r["dataset"] = "cdnet"
    for r in shanghaitech:
        r["dataset"] = "shanghaitech"

    rows = lasiesta + cdnet_train + cdnet_heldout + shanghaitech

    streetscene_path = f"{BASE}/streetscene_rows_with_boundary.csv"
    if os.path.exists(streetscene_path):
        streetscene = cw.read_csv(streetscene_path)
        for r in streetscene:
            r["dataset"] = "streetscene"
        rows = rows + streetscene

    for r in rows:
        r["frame"] = int(float(r["frame"]))
    return rows


def group_key(case):
    # LEGACY (old 90-case, ProPainter-only LASIESTA, superseded 2026-08-09):
    # pristine-only cases used a PRISTINE_ PREFIX instead of a suffix -- strip
    # it to merge with the forged sibling. Only ever matches old frozen CSVs;
    # the new 48-case pristine/propainter/minimax LASIESTA (suffix-based, same
    # convention as every other domain) never hits this branch.
    if case.startswith("PRISTINE_LAS_") or case.startswith("PRISTINE_O_"):
        return case[len("PRISTINE_"):]
    # All 4 domains (including the new LASIESTA, since 2026-08-09) now use a
    # trailing _PRISTINE/_propainter/_minimax suffix -- strip it to get the
    # shared scene/window id, case-insensitively (CDnet uses "_PRISTINE",
    # ShanghaiTech/new-LASIESTA use "_pristine"/"_PRISTINE" respectively).
    return re.sub(r"_(propainter|minimax|pristine)$", "", case, flags=re.IGNORECASE)


# LASIESTA's "MC" (Moving Camera) category: I_MC_01/02, O_MC_01/02, and their
# forged/pristine variants. Criterion (2026-08-04, user-flagged, quantitatively
# confirmed): background-corner-patch phase-correlation shift for I_MC/O_MC is
# 0.19-0.28 px/frame, consistently elevated across BOTH cases in the category,
# vs <0.06 px/frame for every genuinely-static LASIESTA category. This project's
# problem definition is a FIXED/static CCTV camera -- global camera motion of
# this magnitude violates that premise outright, independent of any model's
# performance on these sequences (this is NOT a post-hoc removal of hard cases;
# see project_cfrc_and_motion_branch_status memory for the full derivation and
# the SM-category false-positive check that ruled out crowd-motion artifacts).
# Updated 2026-08-09 for the new 48-case pristine/propainter/minimax LASIESTA
# (suffix-based naming, same convention as every other domain) -- all 12
# case rows below (2 cases x 2 MC ids x 3 generator variants) must be excluded
# from train AND test/val for EVERY model compared (ours and any baseline,
# e.g. Mumpy/ViLocal) -- never test-only, or the comparison is unfair.
MOVING_CAMERA_CASES = {
    "LAS_I_MC_01_PRISTINE", "LAS_I_MC_01_propainter", "LAS_I_MC_01_minimax",
    "LAS_I_MC_02_PRISTINE", "LAS_I_MC_02_propainter", "LAS_I_MC_02_minimax",
    "O_MC_01_PRISTINE", "O_MC_01_propainter", "O_MC_01_minimax",
    "O_MC_02_PRISTINE", "O_MC_02_propainter", "O_MC_02_minimax",
}


def exclude_moving_camera(rows, verbose=True):
    """Fixed-camera subset filter -- see MOVING_CAMERA_CASES docstring above.
    Apply uniformly to train+test+val for every model in a fixed-camera
    comparison; never apply to only one side of a comparison."""
    n_before = len(rows)
    pos_before = sum(1 for r in rows if int(float(r.get("gt_label", r.get("y_true", 0)))) == 1)
    excluded_cases = sorted({r["case"] for r in rows if r["case"] in MOVING_CAMERA_CASES})
    kept = [r for r in rows if r["case"] not in MOVING_CAMERA_CASES]
    if verbose:
        pos_after = sum(1 for r in kept if int(float(r.get("gt_label", r.get("y_true", 0)))) == 1)
        print(f"  [exclude_moving_camera] excluded cases: {excluded_cases}", flush=True)
        print(f"  [exclude_moving_camera] rows {n_before} -> {len(kept)} "
              f"(removed {n_before - len(kept)}); positives {pos_before} -> {pos_after}", flush=True)
    return kept


# =========================================================
# 2. New self-referenced feature families
# =========================================================
def safe_series(case_rows, key):
    return np.asarray([cw.safe_float(r.get(key, 0.0)) for r in case_rows], dtype=np.float64)


def rolling_std(x, win):
    n = len(x)
    half = win // 2
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = np.std(x[lo:hi])
    return out


def rolling_corr(x, y, win):
    n = len(x)
    half = win // 2
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        if hi - lo < 5:
            continue
        xs, ys = x[lo:hi], y[lo:hi]
        if np.std(xs) < 1e-8 or np.std(ys) < 1e-8:
            out[i] = 0.0
            continue
        out[i] = np.corrcoef(xs, ys)[0, 1]
    return out


def add_overconsistency(case_rows):
    for feat in OVERSTABLE_FEATURES:
        x = safe_series(case_rows, feat)
        local_std = rolling_std(x, ROLL_WIN)
        ref_med = np.median(local_std)
        ref_mad = np.median(np.abs(local_std - ref_med)) + 1e-6
        overstable = np.maximum(0.0, (ref_med - local_std) / ref_mad)
        for r, v in zip(case_rows, overstable):
            r[f"{feat}_overstable"] = float(v)


def add_relation_consistency(case_rows):
    for a, b in RELATION_PAIRS:
        xa = safe_series(case_rows, a)
        xb = safe_series(case_rows, b)
        local_corr = rolling_corr(xa, xb, ROLL_WIN)
        valid = local_corr[~np.isnan(local_corr)]
        ref_corr = float(np.median(valid)) if len(valid) else 0.0
        filled = np.where(np.isnan(local_corr), ref_corr, local_corr)
        dev = np.abs(filled - ref_corr)
        key = f"rel_{a}_{b}"
        for r, v in zip(case_rows, dev):
            r[key] = float(v)


def add_temporal_trajectory(case_rows):
    score = safe_series(case_rows, "score")
    transition = safe_series(case_rows, "transition")
    n = len(score)
    accel = np.abs(np.diff(transition, prepend=transition[0])) if n else transition

    med = np.median(score) if n else 0.0
    mad = np.median(np.abs(score - med)) + 1e-6 if n else 1e-6
    thresh = med + 2.0 * mad
    active = score >= thresh

    run_len = np.zeros(n, dtype=np.float64)
    onset = np.zeros(n, dtype=np.float64)
    recovery = np.zeros(n, dtype=np.float64)

    run = 0
    for i in range(n):
        if active[i]:
            if run == 0 and i > 0:
                onset[i] = float(score[i] - score[i - 1])
            run += 1
        else:
            run = 0
        run_len[i] = run

    for i in range(n - 1):
        if active[i] and not active[i + 1]:
            recovery[i] = float(score[i] - score[i + 1])

    for r, rl, on, rec, ac in zip(case_rows, run_len, onset, recovery, accel):
        r["persistence_run_length"] = float(rl)
        r["onset_strength"] = float(on)
        r["recovery_strength"] = float(rec)
        r["accel"] = float(ac)


def add_signed(rows):
    by_case = cw.rows_by_case(rows)
    act_by_case = {c: float(np.median(cw.matrix(r, ["rgb_prev"]).ravel())) for c, r in by_case.items()}
    for r in rows:
        sign = 1.0 if act_by_case[r["case"]] > ACT_THRESH else -1.0
        r["blob_signed"] = float(cw.safe_float(r.get("blob_score", 0.0))) * sign
        r["hp_prev_signed"] = float(cw.safe_float(r.get("hp_prev", 0.0))) * sign
    return rows


# =========================================================
# 3. Stratified group-level 80:20 split
# =========================================================
def stable_fold(dataset, group_id, seed, n_folds=5):
    """Deterministic hash-based fold assignment for ONE (dataset, group_id)
    pair -- independent of any other group's existence, count, or order.
    Fixes a real bug (2026-08-04, found via fixed-camera sensitivity
    analysis): the original shuffle-based split shared one `random.Random`
    across datasets, so `.shuffle()` (which consumes draws proportional to
    list length) let a CHANGE in one dataset's group count silently shift the
    fold assignment of every OTHER dataset too (verified: dropping 8 LASIESTA
    MC cases changed all 27/27 cdnet test-group assignments, despite cdnet
    data being untouched). A per-dataset-only RNG (first fix attempt) still
    wasn't enough: shrinking LASIESTA's own group list would re-shuffle
    LASIESTA's OWN remaining groups differently too. This hash has no
    dependency on list length, order, or what other groups exist -- a given
    (dataset, group_id) always lands in the same fold under a given seed,
    no matter what rows are added or removed elsewhere in the pool."""
    key = f"{seed}::{dataset}::{group_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % n_folds


def _raw_hash_value(dataset, group_id, seed):
    """The full (pre-modulo) hash used by stable_fold(), exposed so the
    minimum-1-test-group fallback below can rank groups deterministically
    without reintroducing any list-order/count dependence."""
    key = f"{seed}::{dataset}::{group_id}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def stratified_split(rows, seed=SEED, verbose=True, n_folds=5):
    """80:20 train/test, group-stratified per dataset, via stable_fold()
    (fold==0 -> test, folds 1..n_folds-1 -> train) -- preserves this
    project's existing 80:20 two-way protocol (no separate validation split;
    inner-OOF calibration is built separately from train_rows where needed),
    but assignment is now fully order/composition-independent per group.

    Minimum-1-test-group guarantee (added 2026-08-04, found via a real
    failure: LASIESTA's small `streetscene`-sibling-scale domains can have
    ALL groups hash to non-zero folds for a given seed by pure chance --
    e.g. streetscene with only 10 groups hit this for seed=1, giving an
    EMPTY test set and a NaN domain AUC). If a dataset's test set comes back
    empty, the single group with the smallest raw hash value (a fixed,
    per-group intrinsic property -- NOT a function of how many other groups
    exist) is deterministically promoted to test. This only ever fires for
    the pathological empty case; it does not touch datasets that already
    got at least one test group honestly."""
    for r in rows:
        r["_group"] = group_key(r["case"])

    groups_by_dataset = defaultdict(set)
    for r in rows:
        groups_by_dataset[r["dataset"]].add(r["_group"])

    train_groups, test_groups = set(), set()
    for ds, groups in groups_by_dataset.items():
        ds_test, ds_train = set(), set()
        for g in groups:
            fold = stable_fold(ds, g, seed, n_folds=n_folds)
            (ds_test if fold == 0 else ds_train).add(g)
        if not ds_test:
            best_g = min(groups, key=lambda g: _raw_hash_value(ds, g, seed))
            ds_test.add(best_g)
            ds_train.discard(best_g)
            print(f"  WARNING [stable_fold]: dataset {ds} got ZERO test groups by hash "
                  f"chance (n_groups={len(groups)}) -- promoted group '{best_g}' "
                  f"(smallest raw hash) to test as a minimum-1 guarantee", flush=True)
        test_groups |= ds_test
        train_groups |= ds_train

    assert train_groups.isdisjoint(test_groups), "leakage: group in both train and test"

    train_rows = [r for r in rows if r["_group"] in train_groups]
    test_rows = [r for r in rows if r["_group"] in test_groups]

    if verbose:
        print(f"=== split summary (seed={seed}, group-level, stratified per dataset) ===")
        for ds, groups in groups_by_dataset.items():
            n_tr = len(groups & train_groups)
            n_te = len(groups & test_groups)
            row_tr = sum(1 for r in train_rows if r["dataset"] == ds)
            row_te = sum(1 for r in test_rows if r["dataset"] == ds)
            print(f"  {ds}: groups train={n_tr} test={n_te} | rows train={row_tr} test={row_te}")
        print(f"  TOTAL: groups train={len(train_groups)} test={len(test_groups)} "
              f"| rows train={len(train_rows)} test={len(test_rows)}")

    return train_rows, test_rows


def group_kfold_splits(rows, k=5, seed=0):
    """Exhaustive group k-fold: every group (scene/window/case) is used as
    test exactly once across the k folds, stratified per dataset (shuffled
    round-robin assignment) so no fold accidentally drops a whole dataset."""
    for r in rows:
        r["_group"] = group_key(r["case"])

    groups_by_dataset = defaultdict(set)
    for r in rows:
        groups_by_dataset[r["dataset"]].add(r["_group"])

    fold_of_group = {}
    rng = random.Random(seed)
    for ds, groups in groups_by_dataset.items():
        gs = sorted(groups)
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            fold_of_group[g] = i % k

    folds = []
    for fold_idx in range(k):
        test_groups = {g for g, f in fold_of_group.items() if f == fold_idx}
        train_groups = {g for g, f in fold_of_group.items() if f != fold_idx}
        assert train_groups.isdisjoint(test_groups)
        train_rows = [r for r in rows if r["_group"] in train_groups]
        test_rows = [r for r in rows if r["_group"] in test_groups]
        folds.append((train_rows, test_rows))
    return folds


# =========================================================
# 4. Design matrix + training
# =========================================================
def build_design(rows, feats, ctx_keys=None):
    x_base = cw.matrix(rows, feats)
    by_case = cw.rows_by_case(rows)
    ctx_by_case = {c: cw.case_context(r, cw.CONTEXT_FEATURES) for c, r in by_case.items()}
    keys = ctx_keys or sorted(next(iter(ctx_by_case.values())).keys())
    c = np.asarray([[ctx_by_case[r["case"]][k] for k in keys] for r in rows], dtype=np.float32)
    return np.concatenate([x_base, c], axis=1), keys


def group_sizes(rows_sorted):
    sizes = []
    cur = None
    for r in rows_sorted:
        if r["_group"] != cur:
            sizes.append(0)
            cur = r["_group"]
        sizes[-1] += 1
    return sizes


def pristine_case_flagging(test_rows, scores, threshold):
    idxs_by_case = defaultdict(list)
    for i, r in enumerate(test_rows):
        idxs_by_case[r["case"]].append(i)

    flagged, total = 0, 0
    for case, idxs in idxs_by_case.items():
        labels = [cw.safe_float(test_rows[i].get("gt_label", 0)) for i in idxs]
        if any(l >= 0.5 for l in labels):
            continue  # not a fully-pristine case
        total += 1
        if any(scores[i] >= threshold for i in idxs):
            flagged += 1
    return flagged, total


def train_and_eval(train_rows, test_rows, feats, label, verbose=True):
    train_sorted = sorted(train_rows, key=lambda r: r["_group"])
    y_train = cw.labels(train_sorted).astype(int)
    X_train, ctx_keys = build_design(train_sorted, feats)
    sizes = group_sizes(train_sorted)

    ranker = lgb.LGBMRanker(objective="lambdarank", verbose=-1, random_state=0, **PARAMS)
    ranker.fit(X_train, y_train, group=sizes)

    X_test, _ = build_design(test_rows, feats, ctx_keys)
    scores = ranker.predict(X_test)
    y_test = cw.labels(test_rows).astype(int)

    auc_by_ds = {}
    for ds in ["lasiesta", "shanghaitech", "cdnet"]:
        mask = np.array([r["dataset"] == ds for r in test_rows])
        if mask.sum() == 0 or len(set(y_test[mask].tolist())) < 2:
            auc_by_ds[ds] = None
        else:
            auc_by_ds[ds] = roc_auc_score(y_test[mask], scores[mask])
    auc_all = roc_auc_score(y_test, scores) if len(set(y_test.tolist())) > 1 else None

    if verbose:
        print(f"\n=== {label} ({len(feats)} base+ctx-independent feats) ===")
        importances = dict(zip(feats, ranker.feature_importances_[:len(feats)]))
        new_feat_names = set(sum(NEW_FEATURE_GROUPS.values(), []))
        top = sorted(importances.items(), key=lambda kv: -kv[1])[:10]
        print("  top-10 feature importances (NEW features marked *):")
        for k, v in top:
            marker = "*" if k in new_feat_names else " "
            print(f"    {marker} {k}: {v}")
        for ds, auc in auc_by_ds.items():
            if auc is None:
                print(f"  AUC[{ds}]: n/a (single class or no rows)")
            else:
                print(f"  AUC[{ds}]: {auc:.4f}")
        if auc_all is not None:
            print(f"  AUC[ALL]: {auc_all:.4f}")

        print("  pct | F1 | precision | recall | frame-FP-rate | pristine-case-flag-rate")
        for pct in (70, 80, 90, 95):
            thresh = np.percentile(scores, pct)
            pred = (scores >= thresh).astype(int)
            tp = int(((pred == 1) & (y_test == 1)).sum())
            fp = int(((pred == 1) & (y_test == 0)).sum())
            fn = int(((pred == 0) & (y_test == 1)).sum())
            tn = int(((pred == 0) & (y_test == 0)).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            fp_rate = fp / max(fp + tn, 1)
            flagged, total = pristine_case_flagging(test_rows, scores, thresh)
            pristine_rate = flagged / max(total, 1)
            print(f"  {pct}th | F1={f1:.4f} | P={precision:.4f} | R={recall:.4f} "
                  f"| FP-rate={fp_rate:.4f} | pristine-flag={flagged}/{total} ({pristine_rate:.4f})")

    return {"auc_by_ds": auc_by_ds, "auc_all": auc_all}


# =========================================================
# main
# =========================================================
def main():
    rows = load_pool()
    rows = add_signed(rows)

    by_case = cw.rows_by_case(rows)
    for case, case_rows in by_case.items():
        case_rows.sort(key=lambda r: r["frame"])
        add_overconsistency(case_rows)
        add_relation_consistency(case_rows)
        add_temporal_trajectory(case_rows)

    train_rows, test_rows = stratified_split(rows)

    # boundary_blur_score is EXCLUDED: it requires a ground-truth removal mask
    # (mask.mp4) to compare in-mask vs. ring sharpness, which isn't available
    # for a genuinely blind/unknown video -- not usable outside this labeled
    # research dataset, so dropped to keep the whole feature set mask-free.
    base_feats = cw.BASE_FEATURES + ["blob_score", "blob_signed", "hp_prev_signed"]

    stages = [
        ("base_context", []),
        ("+overconsistency", NEW_FEATURE_GROUPS["overconsistency"]),
        ("+relation", NEW_FEATURE_GROUPS["relation"]),
        ("+temporal", NEW_FEATURE_GROUPS["temporal"]),
    ]

    cumulative = []
    for name, feats in stages:
        cumulative = cumulative + feats
        train_and_eval(train_rows, test_rows, base_feats + cumulative, name)

    print("\nALL_DONE")


if __name__ == "__main__":
    main()
