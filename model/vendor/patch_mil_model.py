"""
Patch-based MIL (Multiple Instance Learning): addresses the core limitation
of every model built today (LightGBM scalar features, Model A's frame-level
CNN+TCN) -- all of them summarize a WHOLE frame/video into one score, which
dilutes local inpainting signal when only a small region is manipulated.

Design: split each frame into a GRID x GRID spatial grid (native resolution,
so patches keep real local detail -- unlike Model A's single global 32x32
downsample). For each patch, compute the same 4 forensic maps (gray temporal
diff, high-pass, Laplacian, coarse texture), each resized to PATCH_SIZE.
A small shared 2D CNN scores every (patch, frame) instance independently;
the video/window's score is the TOP-K MEAN of all instance scores (standard
MIL aggregation) -- so a single strongly-anomalous patch-frame can drive the
prediction even if the other 95% of the video is pristine, instead of being
averaged away.

Still fully blind at inference: only the video, no GT mask/YOLO/bbox. Video-
level labels only for training (no patch-level supervision) -- this is the
weakly-supervised MIL setup, not a localization-supervised one.

Per today's hard lesson (single-seed "wins" for over-consistency and
reliability-gating both failed to reproduce across seeds): this script
evaluates with multiple CNN-init seeds from the start, never reports a
single-seed number as a finding.
"""
import os
import re
import sys
import time

sys.path.insert(0, "/home/tako/songhye/project4")
sys.path.insert(0, "/home/tako/songhye/project4/scripts_2026_07_29")

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import train_eval_bisfd_pooled_split as bisfd
import forensic_map_encoder_model_a as fmea  # reuse resolve_video_path, blur

# 2026-09-22: see forensic_map_encoder_model_a.extract_maps_for_video's
# docstring -- extract_patch_maps_for_video has the same frame-alignment fix.
FRAME_ALIGN_FIXED = True

GRID = 4                  # 4x4 = 16 patches per frame
PATCH_SIZE = 16
W = 16                     # temporal window, same as Model A
STRIDE_TRAIN = 4
STRIDE_TEST = 2
TOPK_FRAC = 0.10           # top 10% of (patch x frame) instances in a window
BATCH_SIZE = 64            # windows per batch (each window = W*GRID*GRID instances)
EPOCHS = 12
LR = 1e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DOMAINS = ["lasiesta", "shanghaitech", "cdnet", "streetscene"]


def extract_patch_maps_for_video(path, n_frames_needed):
    """See forensic_map_encoder_model_a.extract_maps_for_video's docstring --
    same off-by-one fix: decode one extra frame, drop the artificial
    zero-diff first entry, so index 0 (after the drop) aligns with CSV
    frame=1 and idx = frame - 1 stays correct at every call site."""
    cap = cv2.VideoCapture(path)
    frames_patches = []
    prev_gray = None
    count = 0
    while count < n_frames_needed + 1:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        h, w = gray.shape
        gray_diff = np.zeros_like(gray) if prev_gray is None else np.abs(gray - prev_gray)
        hp = np.abs(gray - fmea.blur(gray, 7))
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        texture = np.abs(gray - fmea.blur(gray, 15))
        full_maps = np.stack([gray_diff, hp, lap, texture], axis=0)  # (4,H,W)

        ph, pw = h // GRID, w // GRID
        patches = np.zeros((GRID * GRID, 4, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        idx = 0
        for gy in range(GRID):
            y0, y1 = gy * ph, (h if gy == GRID - 1 else (gy + 1) * ph)
            for gx in range(GRID):
                x0, x1 = gx * pw, (w if gx == GRID - 1 else (gx + 1) * pw)
                crop = full_maps[:, y0:y1, x0:x1]
                for c in range(4):
                    patches[idx, c] = cv2.resize(crop[c], (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
                idx += 1
        frames_patches.append(patches)
        prev_gray = gray
        count += 1
    cap.release()
    frames_patches = frames_patches[1:]
    if not frames_patches:
        return np.zeros((0, GRID * GRID, 4, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    return np.stack(frames_patches, axis=0)  # (n_frames, 16, 4, 16, 16)


def build_case_patch_maps(rows):
    by_case = bisfd.cw.rows_by_case(rows)
    case_data = {}
    t0 = time.time()
    n_done = 0
    for case, case_rows in by_case.items():
        case_rows.sort(key=lambda r: r["frame"])
        dataset = case_rows[0]["dataset"]
        n_needed = case_rows[-1]["frame"]
        video_path = fmea.resolve_video_path(case, dataset)
        patch_maps = extract_patch_maps_for_video(video_path, n_needed)  # (n,16,4,16,16)
        if patch_maps.shape[0] > 0:
            med = np.median(patch_maps, axis=(0, 1, 3, 4), keepdims=True)
            mad = np.median(np.abs(patch_maps - med), axis=(0, 1, 3, 4), keepdims=True) + 1e-3
            patch_maps = (patch_maps - med) / mad
        labels = np.zeros(patch_maps.shape[0], dtype=np.float32)
        for r in case_rows:
            idx = r["frame"] - 1
            if 0 <= idx < len(labels):
                labels[idx] = bisfd.cw.safe_float(r.get("gt_label", 0.0))
        case_data[case] = (patch_maps, labels, dataset)
        n_done += 1
        if n_done % 20 == 0:
            print(f"    patch-extracted {n_done}/{len(by_case)} cases ({time.time()-t0:.1f}s)", flush=True)
    print(f"  patch map extraction done: {n_done} cases in {time.time()-t0:.1f}s", flush=True)
    return case_data


def make_windows(case_data, stride):
    windows, ys, cases, frames = [], [], [], []
    half = W // 2
    for case, (patch_maps, labels, dataset) in case_data.items():
        n = patch_maps.shape[0]
        if n == 0:
            continue
        padded = np.pad(patch_maps, ((half, half), (0, 0), (0, 0), (0, 0), (0, 0)), mode="edge")
        for t in range(0, n, stride):
            seg = padded[t: t + W]  # (W,16,4,16,16)
            windows.append(seg.astype(np.float32))
            ys.append(labels[t])
            cases.append(case)
            frames.append(t + 1)
    X = np.stack(windows, axis=0)  # (N, W, 16, 4, 16, 16)
    Y = np.asarray(ys, dtype=np.float32)
    return X, Y, cases, frames


class PatchScorer(nn.Module):
    def __init__(self, in_ch=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        # x: (B, in_ch, PATCH_SIZE, PATCH_SIZE) -> (B,) logits
        h = self.net(x).flatten(1)
        return self.head(h).squeeze(-1)


def topk_mean_logits(instance_logits, k_frac):
    # instance_logits: (B, n_instances)
    n = instance_logits.shape[1]
    k = max(1, int(round(n * k_frac)))
    topk, _ = torch.topk(instance_logits, k, dim=1)
    return topk.mean(dim=1)


def train_patch_mil(X_train, Y_train, epochs=EPOCHS):
    model = PatchScorer().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X_train)  # (N, W, 16, 4, 16, 16)
    Yt = torch.from_numpy(Y_train)
    n = Xt.shape[0]

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xt[idx].to(DEVICE)  # (b, W, 16, 4, 16, 16)
            yb = Yt[idx].to(DEVICE)
            b, w_, p_, c_, h_, ww_ = xb.shape
            flat = xb.view(b * w_ * p_, c_, h_, ww_)
            logits_flat = model(flat).view(b, w_ * p_)
            bag_logit = topk_mean_logits(logits_flat, TOPK_FRAC)
            loss = bce(bag_logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        print(f"    epoch {epoch+1}/{epochs} loss={total_loss/n:.4f}", flush=True)
    return model


def predict_patch_mil(model, X_test, batch_size=64):
    model.eval()
    Xt = torch.from_numpy(X_test)
    scores = []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], batch_size):
            xb = Xt[i:i + batch_size].to(DEVICE)
            b, w_, p_, c_, h_, ww_ = xb.shape
            flat = xb.view(b * w_ * p_, c_, h_, ww_)
            logits_flat = model(flat).view(b, w_ * p_)
            bag_logit = topk_mean_logits(logits_flat, TOPK_FRAC)
            scores.append(torch.sigmoid(bag_logit).cpu().numpy())
    return np.concatenate(scores)


FEATS = bisfd.cw.BASE_FEATURES + ["blob_score", "blob_signed", "hp_prev_signed"]


def lgbm_lodo_baseline(train_rows, test_rows, feats):
    for r in train_rows + test_rows:
        r["_group"] = bisfd.group_key(r["case"])
    train_sorted = sorted(train_rows, key=lambda r: r["_group"])
    y_train = bisfd.cw.labels(train_sorted).astype(int)
    X_train, ctx_keys = bisfd.build_design(train_sorted, feats)
    sizes = bisfd.group_sizes(train_sorted)
    ranker = lgb.LGBMRanker(objective="lambdarank", verbose=-1, random_state=0, **bisfd.PARAMS)
    ranker.fit(X_train, y_train, group=sizes)
    X_test, _ = bisfd.build_design(test_rows, feats, ctx_keys)
    scores = ranker.predict(X_test)
    y_test = bisfd.cw.labels(test_rows).astype(int)
    return roc_auc_score(y_test, scores) if len(set(y_test.tolist())) > 1 else None


def run_once(rows, case_data, seed):
    torch.manual_seed(seed)
    results = {}
    for held_out in DOMAINS:
        train_cases = {c: v for c, v in case_data.items() if v[2] != held_out}
        test_cases = {c: v for c, v in case_data.items() if v[2] == held_out}
        train_rows = [r for r in rows if r["dataset"] != held_out]
        test_rows = [r for r in rows if r["dataset"] == held_out]

        X_train, Y_train, _, _ = make_windows(train_cases, stride=STRIDE_TRAIN)
        X_test, Y_test, _, _ = make_windows(test_cases, stride=STRIDE_TEST)
        print(f"  [seed={seed}] held_out={held_out} train_windows={X_train.shape} test_windows={X_test.shape}", flush=True)

        model = train_patch_mil(X_train, Y_train)
        scores = predict_patch_mil(model, X_test)
        auc_mil = roc_auc_score(Y_test, scores) if len(set(Y_test.tolist())) > 1 else None

        auc_lgbm = lgbm_lodo_baseline(train_rows, test_rows, FEATS)
        print(f"  [seed={seed}] held_out={held_out}: PatchMIL={auc_mil:.4f} LightGBM={auc_lgbm:.4f}", flush=True)
        results[held_out] = {"patch_mil": auc_mil, "lgbm": auc_lgbm}
    return results


def main():
    rows = bisfd.load_pool()
    rows = bisfd.add_signed(rows)

    print("extracting PATCH forensic maps for ALL cases (one-time, cached)...", flush=True)
    case_data = build_case_patch_maps(rows)

    SEEDS = [0, 1, 2]
    per_seed = []
    for seed in SEEDS:
        print(f"\n########## seed={seed} ##########", flush=True)
        per_seed.append(run_once(rows, case_data, seed))

    print("\n=== Patch-MIL LODO results across seeds ===")
    for d in DOMAINS:
        mil_vals = [r[d]["patch_mil"] for r in per_seed]
        lgbm_vals = [r[d]["lgbm"] for r in per_seed]
        print(f"  {d}: PatchMIL={np.mean(mil_vals):.4f}+-{np.std(mil_vals):.4f} "
              f"LightGBM={np.mean(lgbm_vals):.4f}+-{np.std(lgbm_vals):.4f}")

    macro_mil = np.mean([np.mean([r[d]["patch_mil"] for d in DOMAINS]) for r in per_seed])
    macro_lgbm = np.mean([np.mean([r[d]["lgbm"] for d in DOMAINS]) for r in per_seed])
    worst_mil = np.mean([min(r[d]["patch_mil"] for d in DOMAINS) for r in per_seed])
    worst_lgbm = np.mean([min(r[d]["lgbm"] for d in DOMAINS) for r in per_seed])
    print(f"\nmean macro LODO AUC across seeds: PatchMIL={macro_mil:.4f} LightGBM={macro_lgbm:.4f}")
    print(f"mean worst-domain LODO AUC across seeds: PatchMIL={worst_mil:.4f} LightGBM={worst_lgbm:.4f}")

    print("\nALL_DONE")


if __name__ == "__main__":
    main()
