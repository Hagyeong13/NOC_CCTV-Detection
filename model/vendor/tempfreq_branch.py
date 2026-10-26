"""TempFreq branch: zero-shot, few-shot fine-tuning, and the activity-input
variant (+ext -- rgb_prev fed as a learned input instead of a hard-coded
post-hoc threshold). Ported against this project's shared
temporal_frequency_model module directly (frame-alignment-fixed)."""
import sys

sys.path.insert(0, "src")
import paths  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import forensic_map_encoder_model_a as fmea  # noqa: E402
import temporal_frequency_model as tfm  # noqa: E402

fmea.resolve_video_path = paths.resolve_video_path


def build_all_case_maps(rows):
    return tfm.build_case_patch_signals(rows)


def subset_case_data(case_data, cases):
    return {c: v for c, v in case_data.items() if c in cases}


def build_rgb_prev_lookup(rows):
    return {(r["case"], int(r["frame"])): float(r.get("rgb_prev", 0.0)) for r in rows}


def rgb_prev_stats(rows, lookup=None):
    lookup = lookup or build_rgb_prev_lookup(rows)
    vals = np.array([lookup.get((r["case"], int(r["frame"])), 0.0) for r in rows], dtype=np.float32)
    return float(vals.mean()), float(vals.std() + 1e-6)


def _augment_activity(X, cases, frames, lookup, mean, std):
    act = np.array([lookup.get((c, int(f)), mean) for c, f in zip(cases, frames)], dtype=np.float32)
    act_z = ((act - mean) / std).astype(np.float32)
    act_b = np.broadcast_to(act_z[:, None, None], (X.shape[0], X.shape[1], 1))
    return np.concatenate([X, act_b], axis=2).astype(np.float32)


def dense_scores_by_row(model, case_data_subset, rows_in_order, activity=None):
    """activity, if given, is (lookup, mean, std) -- augments every window
    with the z-scored rgb_prev activity signal (the +ext configuration)."""
    X, Y, cases, frames = tfm.make_windows(case_data_subset, stride=1)
    if activity is not None:
        X = _augment_activity(X, cases, frames, *activity)
    scores = tfm.predict_freq_model(model, X)
    score_by_key = {(c, f): s for c, f, s in zip(cases, frames, scores)}
    y_by_key = {(c, f): y for c, f, y in zip(cases, frames, Y)}
    scores_rows = np.array([score_by_key[(r["case"], int(r["frame"]))] for r in rows_in_order])
    y_rows = np.array([y_by_key[(r["case"], int(r["frame"]))] for r in rows_in_order])
    return scores_rows, y_rows


def finetune(model, X_adapt, Y_adapt, seed, epochs, lr):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    Xt, Yt = torch.from_numpy(X_adapt), torch.from_numpy(Y_adapt)
    n = Xt.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, tfm.BATCH_SIZE):
            idx = perm[i:i + tfm.BATCH_SIZE]
            xb, yb = Xt[idx].to(tfm.DEVICE), Yt[idx].to(tfm.DEVICE)
            b, p_, d_ = xb.shape
            logits = model(xb.view(b * p_, d_)).view(b, p_)
            bag_logit = tfm.topk_mean_logits(logits, tfm.TOPK_FRAC)
            loss = bce(bag_logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(idx)
        print(f"    [tempfreq finetune] epoch {epoch+1}/{epochs} loss={total_loss/n:.4f}", flush=True)
    return model


def run_zero_shot_and_finetune(train_cd, adapt_cd, test_cd, test_rows, seed, finetune_epochs, finetune_lr):
    """Plain (no activity input) -- used for the `base` config."""
    torch.manual_seed(seed)
    X_train, Y_train, _, _ = tfm.make_windows(train_cd, stride=tfm.STRIDE_TRAIN)
    model = tfm.train_freq_model(X_train, Y_train)
    zero_shot_scores, y_test = dense_scores_by_row(model, test_cd, test_rows)

    X_adapt, Y_adapt, _, _ = tfm.make_windows(adapt_cd, stride=tfm.STRIDE_TRAIN)
    model = finetune(model, X_adapt, Y_adapt, seed, finetune_epochs, finetune_lr)
    finetuned_scores, _ = dense_scores_by_row(model, test_cd, test_rows)
    return zero_shot_scores, finetuned_scores, y_test


def run_finetune_activity(train_cd, adapt_cd, test_cd, test_rows, seed, finetune_epochs, finetune_lr,
                           lookup, mean, std):
    """+ext: rgb_prev fed as a learned input. Used for the `activity_ext`
    config only -- this is what the DA package keeps in its deployed
    configuration (removes the hard-coded threshold=2.7 sign-flip)."""
    torch.manual_seed(seed)
    activity = (lookup, mean, std)
    X_train, Y_train, c_tr, f_tr = tfm.make_windows(train_cd, stride=tfm.STRIDE_TRAIN)
    X_train = _augment_activity(X_train, c_tr, f_tr, *activity)
    model = tfm.train_freq_model(X_train, Y_train)

    X_adapt, Y_adapt, c_ad, f_ad = tfm.make_windows(adapt_cd, stride=tfm.STRIDE_TRAIN)
    X_adapt = _augment_activity(X_adapt, c_ad, f_ad, *activity)
    model = finetune(model, X_adapt, Y_adapt, seed, finetune_epochs, finetune_lr)

    scores, y_test = dense_scores_by_row(model, test_cd, test_rows, activity=activity)
    return scores, y_test
