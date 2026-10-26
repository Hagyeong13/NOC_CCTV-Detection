"""Score-fusion + metric helpers for the unified pipeline. Reuses
dcr_net_full_oof's causal_rolling_mean / recall_at_fpr / partial_auc
(the project's canonical definitions) rather than redefining them."""
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score

import paths  # noqa: F401
import dcr_net_full_oof as dcr

K_SMOOTH = 32
TARGET_FPR = 0.10

causal_rolling_mean = dcr.causal_rolling_mean
recall_at_fpr = dcr.recall_at_fpr
partial_auc = dcr.partial_auc
percentile_rank = dcr.percentile_rank


def source_cdf_transform(reference, new_values):
    """Non-transductive calibration: rank new_values against a FROZEN
    reference distribution (built from train-only OOF scores), never against
    the set being scored. This is the fusion rule for BOTH protocols in this
    release -- see README's "Fusion" section for why the transductive
    alternative (rank-within-the-scored-set) was not carried forward."""
    sorted_ref = np.sort(reference)
    ranks = np.searchsorted(sorted_ref, new_values, side="right")
    return ranks / max(len(sorted_ref), 1)


def full_metrics(y, s):
    """AUC/Recall@10%FPR/pAUC(.1) on the full set, plus F1/recall on a
    class-balanced subset (all positives + equal-count undersampled
    negatives, fixed seed=42) -- convention shared with every prior
    evaluation script in this project."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s)
    rng_bal = np.random.RandomState(42)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        n_take = min(len(pos_idx), len(neg_idx))
        pos_sel = rng_bal.choice(pos_idx, n_take, replace=False) if len(pos_idx) > n_take else pos_idx
        neg_sel = rng_bal.choice(neg_idx, n_take, replace=False) if len(neg_idx) > n_take else neg_idx
        bal_idx = np.concatenate([pos_sel, neg_sel])
        pred_bal = (s[bal_idx] >= 0.5).astype(int)
        f1_pct = float(f1_score(y[bal_idx], pred_bal, zero_division=0)) * 100
        recall_pct = float(recall_score(y[bal_idx], pred_bal, zero_division=0)) * 100
    else:
        f1_pct = float("nan")
        recall_pct = float("nan")
    return {
        "auc": float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan"),
        "recall_at_.1fpr": recall_at_fpr(y, s, TARGET_FPR),
        "pauc_.1": partial_auc(y, s, 0.1),
        "f1_pct": f1_pct, "recall_pct": recall_pct,
    }


def aggregate_by_domain(per_domain_metric, domains):
    """dict[domain]->value for one metric, across the domains actually
    present -> (macro mean, worst/min)."""
    vals = [per_domain_metric[d] for d in domains if d in per_domain_metric]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.min(vals))
