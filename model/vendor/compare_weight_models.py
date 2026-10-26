import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


EPS = 1e-8

BASE_FEATURES = [
    "transition",
    "stable_score",
    "texture_loss",
    "raw_score",
    "rgb_prev",
    "gray_prev",
    "hp_prev",
    "edge_prev",
    "lap_prev",
    "stable",
    "hp_low",
    "lap_low",
    "dct_low",
    "cooccur",
]

REDUCED_FEATURES = [
    "lap_prev",
    "dct_low",
    "edge_prev",
    "raw_score",
    "gray_prev",
    "texture_loss",
    "stable",
    "cooccur",
]

COMPACT_FEATURES = [
    "lap_prev",
    "dct_low",
    "edge_prev",
    "raw_score",
]

CONTEXT_FEATURES = [
    "raw_score",
    "transition",
    "texture_loss",
    "rgb_prev",
    "hp_prev",
    "edge_prev",
    "lap_prev",
    "hp_low",
    "lap_low",
    "dct_low",
]


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def safe_float(value, default=0.0):
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    ensure_dir(os.path.dirname(path))
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def rows_by_case(rows):
    out = defaultdict(list)
    for row in rows:
        out[row["case"]].append(row)
    return out


def matrix(rows, features):
    return np.asarray(
        [[safe_float(row.get(feature, 0.0)) for feature in features] for row in rows],
        dtype=np.float32,
    )


def labels(rows):
    return np.asarray(
        [1 if safe_float(row.get("gt_label", 0.0)) >= 0.5 else 0 for row in rows],
        dtype=np.float32,
    )


def robust_center_scale(x):
    x = np.asarray(x, dtype=np.float32)
    center = np.median(x, axis=0)
    scale = np.median(np.abs(x - center), axis=0)
    fallback = np.std(x, axis=0)
    scale = np.where(scale < EPS, fallback, scale)
    scale = np.where(scale < EPS, 1.0, scale)
    return center.astype(np.float32), scale.astype(np.float32)


def robust_transform(x, center, scale, clip=8.0):
    return np.clip((x - center) / np.maximum(scale, EPS), -clip, clip).astype(np.float32)


def case_context(case_rows, context_features):
    out = {}
    for feature in context_features:
        vals = np.asarray([safe_float(row.get(feature, 0.0)) for row in case_rows], dtype=np.float32)
        out[f"{feature}_mean"] = float(np.mean(vals))
        out[f"{feature}_std"] = float(np.std(vals))
        out[f"{feature}_p90"] = float(np.percentile(vals, 90))
    return out


def context_matrix(rows, context_by_case, context_keys):
    return np.asarray(
        [[context_by_case[row["case"]][key] for key in context_keys] for row in rows],
        dtype=np.float32,
    )


def build_design(train_rows, test_rows, base_features, mode):
    x_train_raw = matrix(train_rows, base_features)
    x_test_raw = matrix(test_rows, base_features)
    base_center, base_scale = robust_center_scale(x_train_raw)
    x_train_base = robust_transform(x_train_raw, base_center, base_scale)
    x_test_base = robust_transform(x_test_raw, base_center, base_scale)

    if mode == "base":
        x_train = x_train_base
        x_test = x_test_base
    else:
        all_rows_by_case = rows_by_case(train_rows + test_rows)
        context_by_case = {
            case: case_context(case_rows, CONTEXT_FEATURES)
            for case, case_rows in all_rows_by_case.items()
        }
        train_cases = sorted(set(row["case"] for row in train_rows))
        context_keys = sorted(next(iter(context_by_case.values())).keys())
        context_train_case_rows = [
            {"case": case, **context_by_case[case]}
            for case in train_cases
        ]
        ctx_train_case = np.asarray(
            [[row[key] for key in context_keys] for row in context_train_case_rows],
            dtype=np.float32,
        )
        ctx_center, ctx_scale = robust_center_scale(ctx_train_case)

        c_train_raw = context_matrix(train_rows, context_by_case, context_keys)
        c_test_raw = context_matrix(test_rows, context_by_case, context_keys)
        c_train = robust_transform(c_train_raw, ctx_center, ctx_scale)
        c_test = robust_transform(c_test_raw, ctx_center, ctx_scale)

        if mode == "context_additive":
            x_train = np.concatenate([x_train_base, c_train], axis=1)
            x_test = np.concatenate([x_test_base, c_test], axis=1)
        elif mode == "context_interaction":
            train_inter = np.einsum("bi,bj->bij", x_train_base, c_train).reshape(len(train_rows), -1)
            test_inter = np.einsum("bi,bj->bij", x_test_base, c_test).reshape(len(test_rows), -1)
            x_train = np.concatenate([x_train_base, c_train, train_inter], axis=1)
            x_test = np.concatenate([x_test_base, c_test, test_inter], axis=1)
        else:
            raise ValueError(f"Unknown design mode: {mode}")

    design_center, design_scale = robust_center_scale(x_train)
    x_train = robust_transform(x_train, design_center, design_scale)
    x_test = robust_transform(x_test, design_center, design_scale)
    return x_train, x_test


def fit_ridge_scores(x_train, y_train, x_test, l2=4.0):
    x_aug = np.concatenate([np.ones((x_train.shape[0], 1), dtype=np.float32), x_train], axis=1)
    x_test_aug = np.concatenate([np.ones((x_test.shape[0], 1), dtype=np.float32), x_test], axis=1)

    pos = max(float(np.sum(y_train == 1)), 1.0)
    neg = max(float(np.sum(y_train == 0)), 1.0)
    sample_weight = np.where(y_train > 0.5, (pos + neg) / (2.0 * pos), (pos + neg) / (2.0 * neg))
    sw = np.sqrt(sample_weight).astype(np.float32)

    xw = x_aug * sw[:, None]
    yw = y_train * sw
    penalty = np.eye(x_aug.shape[1], dtype=np.float32) * float(l2)
    penalty[0, 0] = 0.0

    lhs = xw.T @ xw + penalty
    rhs = xw.T @ yw
    try:
        weights = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return x_aug @ weights, x_test_aug @ weights


def fit_gaussian_nb_scores(x_train, y_train, x_test):
    pos = x_train[y_train > 0.5]
    neg = x_train[y_train <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return np.zeros(len(x_train)), np.zeros(len(x_test))

    mu_pos = np.mean(pos, axis=0)
    mu_neg = np.mean(neg, axis=0)
    var_pos = np.var(pos, axis=0) + 1e-3
    var_neg = np.var(neg, axis=0) + 1e-3
    prior_pos = len(pos) / max(len(x_train), 1)
    prior_neg = len(neg) / max(len(x_train), 1)

    def log_ratio(x):
        lp = -0.5 * np.sum(np.log(var_pos) + ((x - mu_pos) ** 2) / var_pos, axis=1) + math.log(prior_pos + EPS)
        ln = -0.5 * np.sum(np.log(var_neg) + ((x - mu_neg) ** 2) / var_neg, axis=1) + math.log(prior_neg + EPS)
        return lp - ln

    return log_ratio(x_train), log_ratio(x_test)


def fit_centroid_scores(x_train, y_train, x_test):
    pos = x_train[y_train > 0.5]
    neg = x_train[y_train <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return np.zeros(len(x_train)), np.zeros(len(x_test))

    center_pos = np.mean(pos, axis=0)
    center_neg = np.mean(neg, axis=0)

    def score(x):
        d_pos = np.mean((x - center_pos) ** 2, axis=1)
        d_neg = np.mean((x - center_neg) ** 2, axis=1)
        return d_neg - d_pos

    return score(x_train), score(x_test)


def fit_random_forest_scores(x_train, y_train, x_test, args):
    import cv2

    labels_int = y_train.astype(np.int32)
    model = cv2.ml.RTrees_create()
    model.setMaxDepth(args.rf_max_depth)
    model.setMinSampleCount(args.rf_min_sample_count)
    model.setCalculateVarImportance(False)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, args.rf_trees, 0))
    model.train(cv2.ml.TrainData_create(x_train.astype(np.float32), cv2.ml.ROW_SAMPLE, labels_int))

    def score(x):
        try:
            votes = model.getVotes(x.astype(np.float32), 0)
            if votes.shape[0] == x.shape[0] + 1:
                class_labels = votes[0].astype(np.int32).tolist()
                vote_counts = votes[1:].astype(np.float32)
                pos_idx = class_labels.index(1) if 1 in class_labels else vote_counts.shape[1] - 1
            else:
                vote_counts = votes.astype(np.float32)
                pos_idx = min(1, vote_counts.shape[1] - 1)
            total = np.sum(vote_counts, axis=1) + EPS
            return vote_counts[:, pos_idx] / total
        except Exception:
            _, pred = model.predict(x.astype(np.float32))
            return pred.reshape(-1).astype(np.float32)

    return score(x_train), score(x_test)


def fit_hist_gb_scores(x_train, y_train, x_test, args):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=args.gb_max_iter,
        max_depth=args.gb_max_depth,
        learning_rate=args.gb_learning_rate,
        l2_regularization=args.gb_l2,
        random_state=0,
    )
    model.fit(x_train, y_train.astype(np.int32))
    train_score = model.predict_proba(x_train)[:, 1].astype(np.float32)
    test_score = model.predict_proba(x_test)[:, 1].astype(np.float32)
    return train_score, test_score


def fit_svm_scores(x_train, y_train, x_test, args, kernel):
    import cv2

    labels_int = y_train.astype(np.int32)
    train_x = x_train.astype(np.float32)
    train_y = labels_int

    max_train = args.svm_rbf_max_train if kernel == cv2.ml.SVM_RBF else args.svm_linear_max_train
    if max_train > 0 and len(x_train) > max_train:
        rng = np.random.default_rng(args.svm_sample_seed)
        pos_idx = np.where(y_train > 0.5)[0]
        neg_idx = np.where(y_train <= 0.5)[0]
        half = max(1, max_train // 2)
        pos_take = min(len(pos_idx), half)
        neg_take = min(len(neg_idx), max_train - pos_take)
        chosen = np.concatenate([
            rng.choice(pos_idx, size=pos_take, replace=False),
            rng.choice(neg_idx, size=neg_take, replace=False),
        ])
        rng.shuffle(chosen)
        train_x = x_train[chosen].astype(np.float32)
        train_y = labels_int[chosen]

    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(kernel)
    model.setC(args.svm_c)
    if kernel == cv2.ml.SVM_RBF:
        model.setGamma(args.svm_gamma)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, args.svm_max_iter, 1e-4))
    model.train(train_x, cv2.ml.ROW_SAMPLE, train_y)

    def raw_score(x):
        try:
            _, raw = model.predict(x.astype(np.float32), flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            return raw.reshape(-1).astype(np.float32)
        except Exception:
            _, pred = model.predict(x.astype(np.float32))
            return pred.reshape(-1).astype(np.float32)

    train_score = raw_score(x_train)
    test_score = raw_score(x_test)
    return orient_scores(y_train, train_score, test_score)


def fit_stump_boost_scores(x_train, y_train, x_test, args):
    y_pm = np.where(y_train > 0.5, 1.0, -1.0).astype(np.float32)
    n, d = x_train.shape
    if n == 0 or d == 0:
        return np.zeros(n), np.zeros(len(x_test))

    pos = max(float(np.sum(y_train > 0.5)), 1.0)
    neg = max(float(np.sum(y_train <= 0.5)), 1.0)
    weights = np.where(y_train > 0.5, (pos + neg) / (2.0 * pos), (pos + neg) / (2.0 * neg))
    weights = weights.astype(np.float64)
    weights /= np.sum(weights)

    learners = []
    quantiles = np.linspace(0.08, 0.92, args.boost_bins)

    for _ in range(args.boost_estimators):
        best = None
        for feature_idx in range(d):
            values = x_train[:, feature_idx]
            thresholds = np.unique(np.quantile(values, quantiles))
            for threshold in thresholds:
                base_pred = np.where(values >= threshold, 1.0, -1.0)
                for polarity in (1.0, -1.0):
                    pred = base_pred * polarity
                    err = float(np.sum(weights[pred != y_pm]))
                    if best is None or err < best[0]:
                        best = (err, feature_idx, float(threshold), polarity)

        if best is None:
            break

        err, feature_idx, threshold, polarity = best
        err = min(max(err, 1e-6), 1.0 - 1e-6)
        alpha = 0.5 * math.log((1.0 - err) / err)
        pred = np.where(x_train[:, feature_idx] >= threshold, 1.0, -1.0) * polarity
        weights *= np.exp(-alpha * y_pm * pred)
        weights /= np.sum(weights)
        learners.append((alpha, feature_idx, threshold, polarity))

    def score(x):
        out = np.zeros(len(x), dtype=np.float32)
        for alpha, feature_idx, threshold, polarity in learners:
            out += alpha * np.where(x[:, feature_idx] >= threshold, 1.0, -1.0) * polarity
        return out

    return score(x_train), score(x_test)


def fit_mlp_scores(x_train, y_train, x_test, args):
    rng = np.random.default_rng(args.mlp_seed)
    n, d = x_train.shape
    h = args.mlp_hidden
    if n == 0 or d == 0:
        return np.zeros(n), np.zeros(len(x_test))

    w1 = rng.normal(0.0, 0.12, size=(d, h)).astype(np.float32)
    b1 = np.zeros(h, dtype=np.float32)
    w2 = rng.normal(0.0, 0.12, size=(h,)).astype(np.float32)
    b2 = np.float32(0.0)

    y = y_train.astype(np.float32)
    pos = max(float(np.sum(y > 0.5)), 1.0)
    neg = max(float(np.sum(y <= 0.5)), 1.0)
    sample_weight = np.where(y > 0.5, (pos + neg) / (2.0 * pos), (pos + neg) / (2.0 * neg)).astype(np.float32)

    for _ in range(args.mlp_epochs):
        hidden = np.tanh(x_train @ w1 + b1)
        logits = hidden @ w2 + b2
        pred = sigmoid(logits).astype(np.float32)
        dz = (pred - y) * sample_weight / max(n, 1)

        grad_w2 = hidden.T @ dz + args.mlp_l2 * w2 / max(n, 1)
        grad_b2 = np.sum(dz)
        dh = dz[:, None] * w2[None, :] * (1.0 - hidden * hidden)
        grad_w1 = x_train.T @ dh + args.mlp_l2 * w1 / max(n, 1)
        grad_b1 = np.sum(dh, axis=0)

        w2 -= args.mlp_lr * grad_w2
        b2 -= args.mlp_lr * grad_b2
        w1 -= args.mlp_lr * grad_w1
        b1 -= args.mlp_lr * grad_b1

    def score(x):
        return (np.tanh(x @ w1 + b1) @ w2 + b2).astype(np.float32)

    return score(x_train), score(x_test)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def calibrate_with_train(train_score, test_score):
    center = np.median(train_score)
    scale = np.median(np.abs(train_score - center))
    if scale < EPS:
        scale = np.std(train_score)
    if scale < EPS:
        scale = 1.0
    return sigmoid((train_score - center) / scale), sigmoid((test_score - center) / scale)


def binary_metrics(y, pred):
    y = np.asarray(y, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int(np.sum(y & pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    tn = int(np.sum(~y & ~pred))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def auc_rank(y, score):
    y = np.asarray(y, dtype=np.int32)
    score = np.asarray(score, dtype=np.float64)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return 0.5

    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    rank = 1
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        ranks[order[i:j]] = avg_rank
        rank += j - i
        i = j

    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / max(pos * neg, 1)


def choose_threshold(y, score, min_threshold=0.05, max_threshold=0.95):
    candidates = np.unique(np.quantile(score, np.linspace(0.02, 0.98, 129)))
    candidates = candidates[(candidates >= min_threshold) & (candidates <= max_threshold)]
    if candidates.size == 0:
        candidates = np.unique(np.quantile(score, np.linspace(0.05, 0.95, 37)))
    if candidates.size == 0:
        return 0.5

    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        metric = binary_metrics(y, score >= threshold)
        if metric["f1"] > best_f1:
            best_f1 = metric["f1"]
            best_threshold = float(threshold)
    return best_threshold


def orient_scores(y_train, train_score, test_score):
    if auc_rank(y_train, train_score) < 0.5:
        return -train_score, -test_score
    return train_score, test_score


def model_scores(model_name, train_rows, test_rows, args):
    y_train = labels(train_rows)
    y_test = labels(test_rows)

    if model_name == "existing_smooth_score":
        train_score = np.asarray([safe_float(row.get("smooth_score", 0.0)) for row in train_rows], dtype=np.float32)
        test_score = np.asarray([safe_float(row.get("smooth_score", 0.0)) for row in test_rows], dtype=np.float32)
        return train_score, test_score, y_train, y_test

    if model_name == "ridge_full":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "base")
        train_score, test_score = fit_ridge_scores(x_train, y_train, x_test, args.l2)
        return train_score, test_score, y_train, y_test

    if model_name == "ridge_reduced":
        x_train, x_test = build_design(train_rows, test_rows, REDUCED_FEATURES, "base")
        train_score, test_score = fit_ridge_scores(x_train, y_train, x_test, args.l2)
        return train_score, test_score, y_train, y_test

    if model_name == "ridge_compact":
        x_train, x_test = build_design(train_rows, test_rows, COMPACT_FEATURES, "base")
        train_score, test_score = fit_ridge_scores(x_train, y_train, x_test, args.l2)
        return train_score, test_score, y_train, y_test

    if model_name == "context_additive_ridge":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "context_additive")
        train_score, test_score = fit_ridge_scores(x_train, y_train, x_test, args.l2)
        return train_score, test_score, y_train, y_test

    if model_name == "context_interaction_ridge":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "context_interaction")
        train_score, test_score = fit_ridge_scores(x_train, y_train, x_test, args.l2)
        return train_score, test_score, y_train, y_test

    if model_name == "gaussian_nb":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "base")
        train_score, test_score = fit_gaussian_nb_scores(x_train, y_train, x_test)
        return train_score, test_score, y_train, y_test

    if model_name == "centroid_distance":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "base")
        train_score, test_score = fit_centroid_scores(x_train, y_train, x_test)
        return train_score, test_score, y_train, y_test

    if model_name == "random_forest":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "base")
        train_score, test_score = fit_random_forest_scores(x_train, y_train, x_test, args)
        return train_score, test_score, y_train, y_test

    if model_name == "random_forest_reduced":
        x_train, x_test = build_design(train_rows, test_rows, REDUCED_FEATURES, "base")
        train_score, test_score = fit_random_forest_scores(x_train, y_train, x_test, args)
        return train_score, test_score, y_train, y_test

    if model_name == "random_forest_context_additive":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "context_additive")
        train_score, test_score = fit_random_forest_scores(x_train, y_train, x_test, args)
        return train_score, test_score, y_train, y_test

    if model_name == "random_forest_context_interaction":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "context_interaction")
        train_score, test_score = fit_random_forest_scores(x_train, y_train, x_test, args)
        return train_score, test_score, y_train, y_test

    if model_name == "hist_gb":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "base")
        train_score, test_score = fit_hist_gb_scores(x_train, y_train, x_test, args)
        return train_score, test_score, y_train, y_test

    if model_name == "hist_gb_context_additive":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "context_additive")
        train_score, test_score = fit_hist_gb_scores(x_train, y_train, x_test, args)
        return train_score, test_score, y_train, y_test

    if model_name == "hist_gb_context_interaction":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "context_interaction")
        train_score, test_score = fit_hist_gb_scores(x_train, y_train, x_test, args)
        return train_score, test_score, y_train, y_test

    if model_name == "svm_linear":
        import cv2
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "base")
        train_score, test_score = fit_svm_scores(x_train, y_train, x_test, args, cv2.ml.SVM_LINEAR)
        return train_score, test_score, y_train, y_test

    if model_name == "svm_rbf":
        import cv2
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "base")
        train_score, test_score = fit_svm_scores(x_train, y_train, x_test, args, cv2.ml.SVM_RBF)
        return train_score, test_score, y_train, y_test

    if model_name == "stump_boost":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "base")
        train_score, test_score = fit_stump_boost_scores(x_train, y_train, x_test, args)
        return train_score, test_score, y_train, y_test

    if model_name == "mlp_shallow":
        x_train, x_test = build_design(train_rows, test_rows, BASE_FEATURES, "base")
        train_score, test_score = fit_mlp_scores(x_train, y_train, x_test, args)
        return train_score, test_score, y_train, y_test

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(model_name, rows, cases, args):
    all_y = []
    all_score = []
    all_pred = []
    case_rows = []

    for test_case in cases:
        train_rows = [row for row in rows if row["case"] != test_case]
        test_rows = [row for row in rows if row["case"] == test_case]
        train_score, test_score, y_train, y_test = model_scores(model_name, train_rows, test_rows, args)
        train_prob, test_prob = calibrate_with_train(train_score, test_score)
        threshold = choose_threshold(y_train, train_prob, args.min_threshold, args.max_threshold)
        pred = test_prob >= threshold
        metric = binary_metrics(y_test, pred)

        case_rows.append({
            "model": model_name,
            "case": test_case,
            "threshold": threshold,
            "auc": auc_rank(y_test, test_prob),
            **metric,
        })
        all_y.extend(y_test.tolist())
        all_score.extend(test_prob.tolist())
        all_pred.extend(pred.tolist())

    global_metric = binary_metrics(np.asarray(all_y), np.asarray(all_pred))
    global_metric["auc"] = auc_rank(np.asarray(all_y), np.asarray(all_score))
    global_metric["model"] = model_name
    global_metric["cases"] = len(cases)
    return global_metric, case_rows


def evaluate_model_all_videos(model_name, rows, cases, args):
    train_score, test_score, y_train, y_test = model_scores(model_name, rows, rows, args)
    train_prob, test_prob = calibrate_with_train(train_score, test_score)
    threshold = choose_threshold(y_train, train_prob, args.min_threshold, args.max_threshold)
    pred = test_prob >= threshold

    case_rows = []
    case_to_indices = defaultdict(list)
    for idx, row in enumerate(rows):
        case_to_indices[row["case"]].append(idx)

    for case in cases:
        idx = np.asarray(case_to_indices[case], dtype=np.int32)
        y_case = y_test[idx]
        score_case = test_prob[idx]
        pred_case = pred[idx]
        metric = binary_metrics(y_case, pred_case)
        case_rows.append({
            "model": model_name,
            "case": case,
            "threshold": threshold,
            "auc": auc_rank(y_case, score_case),
            **metric,
        })

    global_metric = binary_metrics(y_test, pred)
    global_metric["auc"] = auc_rank(y_test, test_prob)
    global_metric["model"] = model_name
    global_metric["cases"] = len(cases)
    return global_metric, case_rows


def evaluate_ensemble(rows, cases, args, components):
    all_y = []
    all_score = []
    all_pred = []
    case_rows = []

    for test_case in cases:
        train_rows = [row for row in rows if row["case"] != test_case]
        test_rows = [row for row in rows if row["case"] == test_case]
        train_probs = []
        test_probs = []
        y_train_ref = None
        y_test_ref = None

        for component in components:
            train_score, test_score, y_train, y_test = model_scores(component, train_rows, test_rows, args)
            train_prob, test_prob = calibrate_with_train(train_score, test_score)
            train_probs.append(train_prob)
            test_probs.append(test_prob)
            y_train_ref = y_train
            y_test_ref = y_test

        train_mean = np.mean(np.stack(train_probs, axis=0), axis=0)
        test_mean = np.mean(np.stack(test_probs, axis=0), axis=0)
        threshold = choose_threshold(y_train_ref, train_mean, args.min_threshold, args.max_threshold)
        pred = test_mean >= threshold
        metric = binary_metrics(y_test_ref, pred)

        case_rows.append({
            "model": "ensemble_mean",
            "case": test_case,
            "threshold": threshold,
            "auc": auc_rank(y_test_ref, test_mean),
            **metric,
        })
        all_y.extend(y_test_ref.tolist())
        all_score.extend(test_mean.tolist())
        all_pred.extend(pred.tolist())

    global_metric = binary_metrics(np.asarray(all_y), np.asarray(all_pred))
    global_metric["auc"] = auc_rank(np.asarray(all_y), np.asarray(all_score))
    global_metric["model"] = "ensemble_mean"
    global_metric["cases"] = len(cases)
    return global_metric, case_rows


def evaluate_ensemble_all_videos(rows, cases, args, components):
    train_probs = []
    test_probs = []
    y_train_ref = None
    y_test_ref = None

    for component in components:
        train_score, test_score, y_train, y_test = model_scores(component, rows, rows, args)
        train_prob, test_prob = calibrate_with_train(train_score, test_score)
        train_probs.append(train_prob)
        test_probs.append(test_prob)
        y_train_ref = y_train
        y_test_ref = y_test

    train_mean = np.mean(np.stack(train_probs, axis=0), axis=0)
    test_mean = np.mean(np.stack(test_probs, axis=0), axis=0)
    threshold = choose_threshold(y_train_ref, train_mean, args.min_threshold, args.max_threshold)
    pred = test_mean >= threshold

    case_rows = []
    case_to_indices = defaultdict(list)
    for idx, row in enumerate(rows):
        case_to_indices[row["case"]].append(idx)

    for case in cases:
        idx = np.asarray(case_to_indices[case], dtype=np.int32)
        y_case = y_test_ref[idx]
        score_case = test_mean[idx]
        pred_case = pred[idx]
        metric = binary_metrics(y_case, pred_case)
        case_rows.append({
            "model": "ensemble_mean",
            "case": case,
            "threshold": threshold,
            "auc": auc_rank(y_case, score_case),
            **metric,
        })

    global_metric = binary_metrics(y_test_ref, pred)
    global_metric["auc"] = auc_rank(y_test_ref, test_mean)
    global_metric["model"] = "ensemble_mean"
    global_metric["cases"] = len(cases)
    return global_metric, case_rows


def percent_fields(row):
    for key in ["precision", "recall", "f1", "accuracy", "auc"]:
        row[f"{key}_percent"] = row[key] * 100.0
    return row


def make_plots(summary_rows, case_rows, out_dir, evaluation_mode="leave_one_case_out"):
    if not HAS_MPL:
        return []

    metric_label = "Held-out F1 (%)"
    title_suffix = "leave-one-case-out"
    if evaluation_mode == "all_videos":
        metric_label = "All-video F1 (%)"
        title_suffix = "trained and tested on all videos"

    paths = []
    ordered = sorted(summary_rows, key=lambda row: row["f1"])
    labels = [row["model"] for row in ordered]
    f1 = [row["f1_percent"] for row in ordered]
    colors = ["#E07A2F" if row["model"] == "ensemble_mean" else "#3B6EA8" for row in ordered]

    fig, ax = plt.subplots(figsize=(10.8, max(5.0, 0.48 * len(labels) + 1.2)))
    y = range(len(labels))
    ax.barh(y, f1, color=colors, height=0.72)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(metric_label)
    ax.set_title(f"Weight model comparison ({title_suffix})")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(f1):
        ax.text(value + 0.25, i, f"{value:.2f}", va="center", ha="left", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "model_comparison_f1.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for row in summary_rows:
        color = "#E07A2F" if row["model"] == "ensemble_mean" else "#3B6EA8"
        ax.scatter(row["recall_percent"], row["precision_percent"], s=90, color=color)
        ax.text(row["recall_percent"] + 0.25, row["precision_percent"] + 0.25, row["model"], fontsize=8)
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title(f"Precision / Recall trade-off ({title_suffix})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(out_dir, "precision_recall_scatter.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    baseline = next(row for row in summary_rows if row["model"] == "existing_smooth_score")
    delta_rows = [row for row in summary_rows if row["model"] != "existing_smooth_score"]
    delta_rows = sorted(delta_rows, key=lambda row: row["f1"] - baseline["f1"])
    labels = [row["model"] for row in delta_rows]
    deltas = [(row["f1"] - baseline["f1"]) * 100.0 for row in delta_rows]
    colors = ["#E07A2F" if d >= 0 else "#8A8F98" for d in deltas]
    fig, ax = plt.subplots(figsize=(10.8, max(4.8, 0.46 * len(labels) + 1.2)))
    y = range(len(labels))
    ax.barh(y, deltas, color=colors, height=0.72)
    ax.axvline(0.0, color="#222222", linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("F1 change vs existing smooth_score baseline (percentage points)")
    ax.set_title(f"Improvement over current score baseline ({title_suffix})")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(deltas):
        ha = "left" if value >= 0 else "right"
        offset = 0.12 if value >= 0 else -0.12
        ax.text(value + offset, i, f"{value:+.2f}", va="center", ha=ha, fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "model_delta_vs_existing.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    best_by_case = {}
    for row in case_rows:
        case = row["case"]
        if case not in best_by_case or row["f1"] > best_by_case[case]["f1"]:
            best_by_case[case] = row
    counts = Counter(row["model"] for row in best_by_case.values())
    labels = list(counts.keys())
    values = [counts[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(labels, values, color="#4A90A4")
    ax.set_ylabel("Cases where model is best")
    ax.set_title(f"Best model by case ({title_suffix})")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = os.path.join(out_dir, "best_model_by_case.png")
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


def write_report(path, summary_rows, plot_paths, evaluation_mode="leave_one_case_out"):
    best = max(summary_rows, key=lambda row: row["f1"])
    baseline = next(row for row in summary_rows if row["model"] == "existing_smooth_score")
    if evaluation_mode == "all_videos":
        evaluation_text = (
            "Evaluation: all-videos application. Each model is trained on all available cases "
            "and then evaluated on all frames from all cases. This is useful for checking final "
            "behavior across the dataset, but it is not a held-out generalization estimate."
        )
    else:
        evaluation_text = (
            "Evaluation: leave-one-case-out. Each model is trained on all other cases and tested on the held-out case."
        )
    lines = [
        "# Weight Model Comparison",
        "",
        evaluation_text,
        "",
        "## Key Result",
        "",
        (
            f"- Best model: `{best['model']}` "
            f"(F1 {best['f1_percent']:.2f}%, precision {best['precision_percent']:.2f}%, recall {best['recall_percent']:.2f}%)"
        ),
        (
            f"- Existing score baseline: F1 {baseline['f1_percent']:.2f}%, "
            f"precision {baseline['precision_percent']:.2f}%, recall {baseline['recall_percent']:.2f}%"
        ),
        f"- Improvement over existing score: {(best['f1'] - baseline['f1']) * 100.0:+.2f} percentage points",
        "",
        "## Model Table",
        "",
        "| model | F1 % | precision % | recall % | AUC % |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in sorted(summary_rows, key=lambda r: r["f1"], reverse=True):
        lines.append(
            f"| {row['model']} | {row['f1_percent']:.2f} | {row['precision_percent']:.2f} | "
            f"{row['recall_percent']:.2f} | {row['auc_percent']:.2f} |"
        )

    lines.extend(["", "## Plots", ""])
    for plot_path in plot_paths:
        lines.append(f"- `{plot_path}`")

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        default="project4/outputs_final_dataset_hysteresis/feature_context_report/frame_feature_training_table.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="project4/outputs_final_dataset_hysteresis/analysis_report/weight_model_comparison",
    )
    parser.add_argument("--l2", type=float, default=4.0)
    parser.add_argument("--min_threshold", type=float, default=0.05)
    parser.add_argument("--max_threshold", type=float, default=0.95)
    parser.add_argument("--rf_trees", type=int, default=120)
    parser.add_argument("--rf_max_depth", type=int, default=8)
    parser.add_argument("--rf_min_sample_count", type=int, default=8)
    parser.add_argument("--gb_max_iter", type=int, default=300)
    parser.add_argument("--gb_max_depth", type=int, default=6)
    parser.add_argument("--gb_learning_rate", type=float, default=0.08)
    parser.add_argument("--gb_l2", type=float, default=0.0)
    parser.add_argument("--svm_c", type=float, default=2.0)
    parser.add_argument("--svm_gamma", type=float, default=0.15)
    parser.add_argument("--svm_max_iter", type=int, default=700)
    parser.add_argument("--svm_linear_max_train", type=int, default=3200)
    parser.add_argument("--svm_rbf_max_train", type=int, default=2400)
    parser.add_argument("--svm_sample_seed", type=int, default=11)
    parser.add_argument("--boost_estimators", type=int, default=32)
    parser.add_argument("--boost_bins", type=int, default=16)
    parser.add_argument("--mlp_hidden", type=int, default=16)
    parser.add_argument("--mlp_epochs", type=int, default=320)
    parser.add_argument("--mlp_lr", type=float, default=0.04)
    parser.add_argument("--mlp_l2", type=float, default=0.2)
    parser.add_argument("--mlp_seed", type=int, default=7)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument(
        "--eval_mode",
        choices=["leave_one_case_out", "all_videos"],
        default="leave_one_case_out",
        help=(
            "leave_one_case_out estimates held-out generalization; all_videos trains on all "
            "cases and evaluates every frame from every case."
        ),
    )
    parser.add_argument("--skip_ensemble", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.input_csv)
    cases = sorted(set(row["case"] for row in rows))
    ensure_dir(args.out_dir)

    default_models = [
        "existing_smooth_score",
        "ridge_compact",
        "ridge_reduced",
        "ridge_full",
        "context_additive_ridge",
        "context_interaction_ridge",
        "gaussian_nb",
        "centroid_distance",
        "random_forest",
        "random_forest_reduced",
        "svm_linear",
        "svm_rbf",
        "stump_boost",
        "mlp_shallow",
    ]
    models = args.models or default_models

    summary_rows = []
    all_case_rows = []

    for model_name in models:
        print("evaluating:", model_name, flush=True)
        if args.eval_mode == "all_videos":
            metric, by_case = evaluate_model_all_videos(model_name, rows, cases, args)
        else:
            metric, by_case = evaluate_model(model_name, rows, cases, args)
        metric = percent_fields(metric)
        metric["eval_mode"] = args.eval_mode
        summary_rows.append(metric)
        for row in by_case:
            row = percent_fields(row)
            row["eval_mode"] = args.eval_mode
            all_case_rows.append(row)

    if not args.skip_ensemble:
        print("evaluating:", "ensemble_mean", flush=True)
        if args.eval_mode == "all_videos":
            ensemble_metric, ensemble_case_rows = evaluate_ensemble_all_videos(
                rows,
                cases,
                args,
                ["ridge_full", "context_additive_ridge", "random_forest", "svm_linear", "mlp_shallow"],
            )
        else:
            ensemble_metric, ensemble_case_rows = evaluate_ensemble(
                rows,
                cases,
                args,
                ["ridge_full", "context_additive_ridge", "random_forest", "svm_linear", "mlp_shallow"],
            )
        ensemble_metric = percent_fields(ensemble_metric)
        ensemble_metric["eval_mode"] = args.eval_mode
        summary_rows.append(ensemble_metric)
        for row in ensemble_case_rows:
            row = percent_fields(row)
            row["eval_mode"] = args.eval_mode
            all_case_rows.append(row)

    summary_rows = sorted(summary_rows, key=lambda row: row["f1"], reverse=True)
    summary_csv = os.path.join(args.out_dir, "model_comparison_summary.csv")
    case_csv = os.path.join(args.out_dir, "model_comparison_by_case.csv")
    json_path = os.path.join(args.out_dir, "model_comparison_summary.json")
    report_path = os.path.join(args.out_dir, "model_comparison_report.md")

    write_csv(summary_csv, summary_rows)
    write_csv(case_csv, all_case_rows)
    save_json(json_path, {"eval_mode": args.eval_mode, "summary": summary_rows})
    plot_paths = make_plots(summary_rows, all_case_rows, args.out_dir, args.eval_mode)
    write_report(report_path, summary_rows, plot_paths, args.eval_mode)

    best = summary_rows[0]
    baseline = next(row for row in summary_rows if row["model"] == "existing_smooth_score")
    print("summary:", summary_csv)
    print("by_case:", case_csv)
    print("report:", report_path)
    for path in plot_paths:
        print("plot:", path)
    print(
        "best:",
        best["model"],
        f"f1={best['f1_percent']:.2f}%",
        f"precision={best['precision_percent']:.2f}%",
        f"recall={best['recall_percent']:.2f}%",
        f"delta_vs_existing={(best['f1'] - baseline['f1']) * 100.0:+.2f}pp",
    )


if __name__ == "__main__":
    main()
