"""The one fusion rule this release uses for BOTH protocols (pooled and
lodo_adapt): source-CDF calibration against a train-only, out-of-fold
reference distribution per branch, then equal-weight average, then causal
K=32 smoothing.

Deliberately NOT the DA package's plain within-scored-set percentile_rank
(transductive -- a frame's fused value depends on which other frames were
scored alongside it). That version existed to stay numerically comparable
with this project's older baselines (dcr_net_full_oof.py-era scripts); this
release is the proposed framework itself, not a comparison against that
lineage, so it uses the calibration the project's own newer pipeline
(pooled_split_5way_extended.py, after the external-review "transductive
fusion" fix) already adopted. See BRANCHES = the 5 branch names fused.
"""
import numpy as np

import metrics

BRANCHES = ["lgbm", "compression", "cnn", "patch", "tempfreq"]


def build_source_cdf_reference(branch_scores_oof):
    """branch_scores_oof: dict[branch] -> 1D array of scores from an inner
    (train-only) group-k-fold, i.e. never touching val/test/adapt rows.
    Returns the same dict, ready to pass to fuse()."""
    return {b: np.asarray(v) for b, v in branch_scores_oof.items()}


def fuse(branch_scores_eval, source_cdf_reference, rows_eval, window=metrics.K_SMOOTH):
    """branch_scores_eval: dict[branch] -> 1D array (same order as rows_eval).
    Calibrates each branch against its OWN frozen reference, averages with
    equal weight, then causally smooths per-case."""
    calibrated = {
        b: metrics.source_cdf_transform(source_cdf_reference[b], branch_scores_eval[b])
        for b in BRANCHES if b in branch_scores_eval
    }
    fused = np.mean(np.stack(list(calibrated.values()), axis=0), axis=0)
    smoothed = metrics.causal_rolling_mean(fused, rows_eval, window)
    return smoothed, calibrated
