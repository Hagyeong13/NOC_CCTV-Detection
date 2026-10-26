"""Model A (2D CNN + TCN) branch: zero-shot scoring and, for activity_ext,
few-shot fine-tuning. Uses forensic_map_encoder_model_a directly (the
project's shared module, frame-alignment-fixed as of 2026-09-22) -- no
vendored copy, no legacy/fixed toggle.

Fine-tune mechanism ported from the DA package's src/cnn_branch.py
(run_finetune_raw_with_base / finetune_model_a): continues gradient descent
on the already-source-trained network (whole network, not just the head) for
a few epochs at a low LR on the labeled adapt-set -- the analog, for a
fixed-depth network, of the LGBM branch's continued-boosting fine-tune."""
import sys

sys.path.insert(0, "src")
import paths  # noqa: F401,E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import forensic_map_encoder_model_a as fmea  # noqa: E402

fmea.resolve_video_path = paths.resolve_video_path


def build_all_case_maps(rows):
    return fmea.build_case_maps(rows)


def subset_case_data(case_data, cases):
    return {c: v for c, v in case_data.items() if c in cases}


def train_zero_shot(train_case_data):
    X_train, Y_train, _, _, _ = fmea.make_windows(train_case_data, stride=fmea.STRIDE_TRAIN)
    return fmea.train_model_a(X_train, Y_train)


def dense_scores_by_row(model, case_data_subset, rows_in_order):
    X, Y, _, cases, frames = fmea.make_windows(case_data_subset, stride=1)
    scores = fmea.predict_model_a(model, X)
    score_by_key = {(c, f): s for c, f, s in zip(cases, frames, scores)}
    y_by_key = {(c, f): y for c, f, y in zip(cases, frames, Y)}
    scores_rows = np.array([score_by_key[(r["case"], int(r["frame"]))] for r in rows_in_order])
    y_rows = np.array([y_by_key[(r["case"], int(r["frame"]))] for r in rows_in_order])
    return scores_rows, y_rows


def finetune(model, X_adapt, Y_adapt, seed, epochs, lr):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()
    Xt, Yt = torch.from_numpy(X_adapt), torch.from_numpy(Y_adapt)
    n = Xt.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, fmea.BATCH_SIZE):
            idx = perm[i:i + fmea.BATCH_SIZE]
            xb, yb = Xt[idx].to(fmea.DEVICE), Yt[idx].to(fmea.DEVICE)
            loss = bce(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(idx)
        print(f"    [cnn finetune] epoch {epoch+1}/{epochs} loss={total_loss/n:.4f}", flush=True)
    return model


def run_zero_shot_and_finetune(train_cd, adapt_cd, test_cd, test_rows, seed, finetune_epochs, finetune_lr):
    """Returns (zero_shot_scores, finetuned_scores, y_test) on test_rows --
    one source-training run serves both the `base` and `activity_ext`
    configurations."""
    torch.manual_seed(seed)
    model = train_zero_shot(train_cd)
    zero_shot_scores, y_test = dense_scores_by_row(model, test_cd, test_rows)

    X_adapt, Y_adapt, _, _, _ = fmea.make_windows(adapt_cd, stride=fmea.STRIDE_TRAIN)
    model = finetune(model, X_adapt, Y_adapt, seed, finetune_epochs, finetune_lr)
    finetuned_scores, _ = dense_scores_by_row(model, test_cd, test_rows)
    return zero_shot_scores, finetuned_scores, y_test
