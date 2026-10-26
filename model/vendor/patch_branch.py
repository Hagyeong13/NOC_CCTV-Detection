"""Patch-MIL branch: zero-shot scoring and few-shot fine-tuning, ported from
the DA package's src/patch_branch.py against this project's shared
patch_mil_model module directly (frame-alignment-fixed, no vendored copy)."""
import sys

sys.path.insert(0, "src")
import paths  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import forensic_map_encoder_model_a as fmea  # noqa: E402
import patch_mil_model as pmm  # noqa: E402

fmea.resolve_video_path = paths.resolve_video_path

MAX_DENSE_FRAMES_PER_CHUNK = 4000  # bounds peak memory when densely scoring a large case set


def build_all_case_maps(rows):
    return pmm.build_case_patch_maps(rows)


def subset_case_data(case_data, cases):
    return {c: v for c, v in case_data.items() if c in cases}


def train_zero_shot(train_case_data):
    X_train, Y_train, _, _ = pmm.make_windows(train_case_data, stride=pmm.STRIDE_TRAIN)
    model = pmm.train_patch_mil(X_train, Y_train)
    del X_train
    return model


def dense_scores_by_row(model, case_data_subset, rows_in_order):
    """Chunked (bounded peak memory) dense scoring over a large case set."""
    score_by_key, y_by_key = {}, {}
    items = list(case_data_subset.items())

    def _flush(chunk):
        if not chunk:
            return
        X, Y, cs, frames = pmm.make_windows(chunk, stride=1)
        scores = pmm.predict_patch_mil(model, X)
        for c, f, s, y in zip(cs, frames, scores, Y):
            score_by_key[(c, f)] = s
            y_by_key[(c, f)] = y

    chunk, chunk_frames = {}, 0
    for case, v in items:
        n_frames = v[0].shape[0]
        if chunk and chunk_frames + n_frames > MAX_DENSE_FRAMES_PER_CHUNK:
            _flush(chunk)
            chunk, chunk_frames = {}, 0
        chunk[case] = v
        chunk_frames += n_frames
    _flush(chunk)

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
        for i in range(0, n, pmm.BATCH_SIZE):
            idx = perm[i:i + pmm.BATCH_SIZE]
            xb, yb = Xt[idx].to(pmm.DEVICE), Yt[idx].to(pmm.DEVICE)
            b, w_, p_, c_, h_, ww_ = xb.shape
            flat = xb.view(b * w_ * p_, c_, h_, ww_)
            logits_flat = model(flat).view(b, w_ * p_)
            bag_logit = pmm.topk_mean_logits(logits_flat, pmm.TOPK_FRAC)
            loss = bce(bag_logit, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(idx)
        print(f"    [patch finetune] epoch {epoch+1}/{epochs} loss={total_loss/n:.4f}", flush=True)
    return model


def run_zero_shot_and_finetune(train_cd, adapt_cd, test_cd, test_rows, seed, finetune_epochs, finetune_lr):
    torch.manual_seed(seed)
    model = train_zero_shot(train_cd)
    zero_shot_scores, y_test = dense_scores_by_row(model, test_cd, test_rows)

    X_adapt, Y_adapt, _, _ = pmm.make_windows(adapt_cd, stride=pmm.STRIDE_TRAIN)
    model = finetune(model, X_adapt, Y_adapt, seed, finetune_epochs, finetune_lr)
    del X_adapt
    finetuned_scores, _ = dense_scores_by_row(model, test_cd, test_rows)
    return zero_shot_scores, finetuned_scores, y_test
