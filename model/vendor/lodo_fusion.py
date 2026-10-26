"""
Test whether LightGBM (scalar forensic features) and Model A (CNN+TCN on
forensic maps) are genuinely complementary, per the LODO result showing
LightGBM strong on LASIESTA / weak on ShanghaiTech+CDnet and Model A the
mirror image. Two fusion methods, both evaluated on the same 3 LODO folds:

  1. Rank/percentile fusion: convert each model's held-out-domain scores to
     percentile ranks (within that same test set, unsupervised, no labels
     used) and average -- no fitting needed, robust to score-scale mismatch.
  2. Source-domain-only stacking: a logistic regression on [LGBM_score,
     CNN_score] -> label, fit ONLY on an inner holdout carved from the 2
     SOURCE domains (never the true held-out domain), then applied to the
     held-out domain's scores. This is the leakage-free requirement the user
     flagged: fusion weights must never be chosen by looking at the domain
     being tested.
"""
import sys

sys.path.insert(0, "/home/tako/songhye/project4/scripts_2026_07_29")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import forensic_map_encoder_model_a as fmea
import train_eval_bisfd_pooled_split as bisfd

DOMAINS = ["lasiesta", "shanghaitech", "cdnet", "streetscene"]
FEATS = fmea.FEATS
INNER_VAL_FRAC = 0.3
INNER_SEED = 777


def percentile_rank(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x))
    return ranks / max(len(x) - 1, 1)


def cnn_scores_by_row(model, case_data_subset, rows_in_order):
    """Score every frame densely (stride=1) and reindex to match rows_in_order's
    exact (case, frame) sequence -- required so CNN and LightGBM scores can be
    fused sample-for-sample (their internal iteration orders otherwise differ)."""
    X, Y, doms, cases, frames = fmea.make_windows(case_data_subset, stride=1)
    preds = fmea.predict_model_a(model, X)
    score_by_key = {(c, f): s for c, f, s in zip(cases, frames, preds)}
    label_by_key = {(c, f): y for c, f, y in zip(cases, frames, Y)}
    scores = np.array([score_by_key[(r["case"], int(r["frame"]))] for r in rows_in_order])
    labels = np.array([label_by_key[(r["case"], int(r["frame"]))] for r in rows_in_order])
    return scores, labels


def lgbm_train_predict(train_rows, test_rows, feats, seed=0):
    for r in train_rows + test_rows:
        r["_group"] = bisfd.group_key(r["case"])
    train_sorted = sorted(train_rows, key=lambda r: r["_group"])
    y_train = bisfd.cw.labels(train_sorted).astype(int)
    X_train, ctx_keys = bisfd.build_design(train_sorted, feats)
    sizes = bisfd.group_sizes(train_sorted)
    ranker = lgb.LGBMRanker(objective="lambdarank", verbose=-1, random_state=seed, **bisfd.PARAMS)
    ranker.fit(X_train, y_train, group=sizes)
    X_test, _ = bisfd.build_design(test_rows, feats, ctx_keys)
    return ranker.predict(X_test)


def inner_source_split(source_cases, seed):
    """Group-stratified split WITHIN the 2 source domains only, used solely
    to fit fusion weights -- never touches the held-out test domain."""
    groups_by_dataset = {}
    for case, v in source_cases.items():
        ds = v[2]
        groups_by_dataset.setdefault(ds, set()).add(bisfd.group_key(case))
    import random
    rng = random.Random(seed)
    val_groups = set()
    for ds, groups in groups_by_dataset.items():
        gs = sorted(groups)
        rng.shuffle(gs)
        n_val = max(1, round(len(gs) * INNER_VAL_FRAC))
        val_groups.update(gs[:n_val])
    inner_train = {c: v for c, v in source_cases.items() if bisfd.group_key(c) not in val_groups}
    inner_val = {c: v for c, v in source_cases.items() if bisfd.group_key(c) in val_groups}
    return inner_train, inner_val


def main():
    rows = bisfd.load_pool()
    rows = bisfd.add_signed(rows)

    print("extracting forensic maps for ALL cases (cached once, reused across folds)...", flush=True)
    case_data = fmea.build_case_maps(rows)

    fold_results = {}
    for held_out in DOMAINS:
        print(f"\n=== LODO: held-out domain = {held_out} ===", flush=True)
        source_cases = {c: v for c, v in case_data.items() if v[2] != held_out}
        test_cases = {c: v for c, v in case_data.items() if v[2] == held_out}
        train_rows = [r for r in rows if r["dataset"] != held_out]
        test_rows = [r for r in rows if r["dataset"] == held_out]

        # ---- full-source models (used for the actual held-out-domain scores) ----
        print("  training full-source LightGBM + Model A...", flush=True)
        lgbm_scores_test = lgbm_train_predict(train_rows, test_rows, FEATS)

        X_train_full, Y_train_full, _, _, _ = fmea.make_windows(source_cases, stride=fmea.STRIDE_TRAIN)
        model_full = fmea.train_model_a(X_train_full, Y_train_full)
        # dense (stride=1) + reindexed to test_rows' exact (case,frame) order, so
        # this lines up 1:1 with lgbm_scores_test/y_test for fusion
        cnn_scores_test, cnn_y_test = cnn_scores_by_row(model_full, test_cases, test_rows)

        y_test = bisfd.cw.labels(test_rows).astype(int)
        assert len(y_test) == len(lgbm_scores_test) == len(cnn_scores_test)
        assert np.array_equal(y_test, cnn_y_test.astype(int))

        auc_lgbm = roc_auc_score(y_test, lgbm_scores_test)
        auc_cnn = roc_auc_score(y_test, cnn_scores_test)

        # ---- rank/percentile fusion (unsupervised, computed on test scores directly) ----
        rank_fused = 0.5 * percentile_rank(lgbm_scores_test) + 0.5 * percentile_rank(cnn_scores_test)
        auc_rank_fusion = roc_auc_score(y_test, rank_fused)

        # ---- source-domain-only stacking ----
        print("  building inner source-only holdout for stacking...", flush=True)
        inner_train_cases, inner_val_cases = inner_source_split(source_cases, seed=INNER_SEED)
        inner_train_rows = [r for c in inner_train_cases for r in bisfd.cw.rows_by_case(train_rows)[c]]
        inner_val_rows = [r for c in inner_val_cases for r in bisfd.cw.rows_by_case(train_rows)[c]]

        print("  training inner-source LightGBM + Model A (for stacking fit only)...", flush=True)
        lgbm_scores_innerval = lgbm_train_predict(inner_train_rows, inner_val_rows, FEATS)

        X_inner_train, Y_inner_train, _, _, _ = fmea.make_windows(inner_train_cases, stride=fmea.STRIDE_TRAIN)
        model_inner = fmea.train_model_a(X_inner_train, Y_inner_train, epochs=fmea.EPOCHS)
        cnn_scores_innerval, cnn_y_innerval = cnn_scores_by_row(model_inner, inner_val_cases, inner_val_rows)

        y_innerval = bisfd.cw.labels(inner_val_rows).astype(int)
        assert len(y_innerval) == len(lgbm_scores_innerval) == len(cnn_scores_innerval)
        assert np.array_equal(y_innerval, cnn_y_innerval.astype(int))

        Xs = np.stack([lgbm_scores_innerval, cnn_scores_innerval], axis=1)
        mean, std = Xs.mean(axis=0), Xs.std(axis=0) + 1e-6
        lr = LogisticRegression()
        lr.fit((Xs - mean) / std, y_innerval)
        print(f"  stacking weights (standardized): {lr.coef_}, intercept={lr.intercept_}", flush=True)

        Xt = np.stack([lgbm_scores_test, cnn_scores_test], axis=1)
        stacked_scores = lr.predict_proba((Xt - mean) / std)[:, 1]
        auc_stack = roc_auc_score(y_test, stacked_scores)

        print(f"  AUC: LightGBM={auc_lgbm:.4f} CNN={auc_cnn:.4f} "
              f"rank_fusion={auc_rank_fusion:.4f} stacking={auc_stack:.4f}", flush=True)

        fold_results[held_out] = {
            "lgbm": auc_lgbm, "cnn": auc_cnn,
            "rank_fusion": auc_rank_fusion, "stacking": auc_stack,
        }

    print("\n=== LODO fusion comparison ===")
    print(f"{'held-out':<16} {'LightGBM':>10} {'CNN':>10} {'rank_fusion':>13} {'stacking':>10}")
    for d in DOMAINS:
        r = fold_results[d]
        print(f"{d:<16} {r['lgbm']:>10.4f} {r['cnn']:>10.4f} {r['rank_fusion']:>13.4f} {r['stacking']:>10.4f}")

    for method in ["lgbm", "cnn", "rank_fusion", "stacking"]:
        vals = [fold_results[d][method] for d in DOMAINS]
        print(f"{method:<16} macro={np.mean(vals):.4f} worst={min(vals):.4f}")

    print("\nALL_DONE")


if __name__ == "__main__":
    main()
