"""
Temporal-Frequency branch (priority #2 from the user's post-fusion plan):
a representation genuinely different from every time-domain feature tried
today (Δ, Δ², multilag, persistence, TCN) -- instead of looking at HOW a
signal changes frame-to-frame, take the FFT of each patch's temporal signal
over a window and look at its frequency-domain structure (periodic/
non-periodic local temporal patterns that time-domain diffs don't expose).

Design: reuse the same 4x4 spatial patch grid as patch_mil_model.py, but per
patch track two cheap scalar time series (mean gray intensity, mean
high-pass-residual magnitude) instead of full 2D maps. For a FFT_WINDOW=32
frame window centered at each frame, take the magnitude spectrum (rfft) of
both series per patch -> a small MLP scores each patch -> Top-K MIL
aggregation across patches (same aggregation as Patch-MIL) -> one score per
frame, aligned to (case, frame) for later fusion with the other 3 branches.
Still fully blind at inference (video only, video-level labels for MIL
training).
"""
import sys
import time

sys.path.insert(0, "/home/tako/songhye/project4")
sys.path.insert(0, "/home/tako/songhye/project4/scripts_2026_07_29")

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import train_eval_bisfd_pooled_split as bisfd
import forensic_map_encoder_model_a as fmea  # resolve_video_path, blur

# 2026-09-22: see forensic_map_encoder_model_a.extract_maps_for_video's
# docstring -- extract_patch_signals_for_video has the same frame-alignment fix.
FRAME_ALIGN_FIXED = True

GRID = 4
FFT_WINDOW = 32       # frames per FFT window (-> 17 magnitude bins via rfft)
STRIDE_TRAIN = 4
STRIDE_TEST = 2
TOPK_FRAC = 0.25       # top 25% of 16 patches (4 patches) -- fewer instances than Patch-MIL's patch*frame bag
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DOMAINS = ["lasiesta", "shanghaitech", "cdnet", "streetscene"]
N_BINS = FFT_WINDOW // 2 + 1


def extract_patch_signals_for_video(path, n_frames_needed):
    """Returns (n_frames, 16, 2): per-patch mean gray intensity and mean
    high-pass-residual magnitude, per frame -- cheap scalars, not full maps.
    Decodes one extra leading frame and drops it (see
    forensic_map_encoder_model_a.extract_maps_for_video's docstring): even
    though this function has no diff to zero-fill, index 0 must still align
    with CSV frame=1 rather than frame=0 for idx = frame - 1 to be correct."""
    cap = cv2.VideoCapture(path)
    signals = []
    count = 0
    while count < n_frames_needed + 1:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        hp = np.abs(gray - fmea.blur(gray, 7))
        h, w = gray.shape
        ph, pw = h // GRID, w // GRID
        patch_vals = np.zeros((GRID * GRID, 2), dtype=np.float32)
        idx = 0
        for gy in range(GRID):
            y0, y1 = gy * ph, (h if gy == GRID - 1 else (gy + 1) * ph)
            for gx in range(GRID):
                x0, x1 = gx * pw, (w if gx == GRID - 1 else (gx + 1) * pw)
                patch_vals[idx, 0] = gray[y0:y1, x0:x1].mean()
                patch_vals[idx, 1] = hp[y0:y1, x0:x1].mean()
                idx += 1
        signals.append(patch_vals)
        count += 1
    cap.release()
    signals = signals[1:]
    if not signals:
        return np.zeros((0, GRID * GRID, 2), dtype=np.float32)
    return np.stack(signals, axis=0)  # (n_frames, 16, 2)


def build_case_patch_signals(rows):
    by_case = bisfd.cw.rows_by_case(rows)
    case_data = {}
    t0 = time.time()
    n_done = 0
    for case, case_rows in by_case.items():
        case_rows.sort(key=lambda r: r["frame"])
        dataset = case_rows[0]["dataset"]
        n_needed = case_rows[-1]["frame"]
        video_path = fmea.resolve_video_path(case, dataset)
        sig = extract_patch_signals_for_video(video_path, n_needed)  # (n,16,2)
        labels = np.zeros(sig.shape[0], dtype=np.float32)
        for r in case_rows:
            idx = r["frame"] - 1
            if 0 <= idx < len(labels):
                labels[idx] = bisfd.cw.safe_float(r.get("gt_label", 0.0))
        case_data[case] = (sig, labels, dataset)
        n_done += 1
        if n_done % 40 == 0:
            print(f"    signal-extracted {n_done}/{len(by_case)} cases ({time.time()-t0:.1f}s)", flush=True)
    print(f"  patch-signal extraction done: {n_done} cases in {time.time()-t0:.1f}s", flush=True)
    return case_data


def fft_features_for_window(sig_window):
    """sig_window: (FFT_WINDOW, 16, 2) -> (16, 2*N_BINS) magnitude spectra,
    normalized by window's own DC-removed std (self-referenced per window,
    consistent with this project's convention)."""
    x = sig_window - sig_window.mean(axis=0, keepdims=True)  # remove DC per patch/channel
    spec = np.abs(np.fft.rfft(x, axis=0))  # (N_BINS, 16, 2)
    spec = spec.transpose(1, 0, 2).reshape(GRID * GRID, -1)  # (16, N_BINS*2)
    norm = spec.std(axis=1, keepdims=True) + 1e-6
    return (spec / norm).astype(np.float32)


def make_windows(case_data, stride):
    windows, ys, cases, frames = [], [], [], []
    half = FFT_WINDOW // 2
    for case, (sig, labels, dataset) in case_data.items():
        n = sig.shape[0]
        if n == 0:
            continue
        padded = np.pad(sig, ((half, half), (0, 0), (0, 0)), mode="edge")
        for t in range(0, n, stride):
            seg = padded[t: t + FFT_WINDOW]  # (FFT_WINDOW,16,2)
            feat = fft_features_for_window(seg)  # (16, 2*N_BINS)
            windows.append(feat)
            ys.append(labels[t])
            cases.append(case)
            frames.append(t + 1)
    X = np.stack(windows, axis=0)  # (N, 16, 2*N_BINS)
    Y = np.asarray(ys, dtype=np.float32)
    return X, Y, cases, frames


class FreqScorer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def topk_mean_logits(instance_logits, k_frac):
    n = instance_logits.shape[1]
    k = max(1, int(round(n * k_frac)))
    topk, _ = torch.topk(instance_logits, k, dim=1)
    return topk.mean(dim=1)


def train_freq_model(X_train, Y_train, epochs=EPOCHS):
    in_dim = X_train.shape[-1]
    model = FreqScorer(in_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X_train)
    Yt = torch.from_numpy(Y_train)
    n = Xt.shape[0]

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb = Xt[idx].to(DEVICE)  # (b, 16, in_dim)
            yb = Yt[idx].to(DEVICE)
            b, p_, d_ = xb.shape
            logits = model(xb.view(b * p_, d_)).view(b, p_)
            bag_logit = topk_mean_logits(logits, TOPK_FRAC)
            loss = bce(bag_logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        print(f"    epoch {epoch+1}/{epochs} loss={total_loss/n:.4f}", flush=True)
    return model


def predict_freq_model(model, X_test, batch_size=512):
    model.eval()
    Xt = torch.from_numpy(X_test)
    scores = []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], batch_size):
            xb = Xt[i:i + batch_size].to(DEVICE)
            b, p_, d_ = xb.shape
            logits = model(xb.view(b * p_, d_)).view(b, p_)
            bag_logit = topk_mean_logits(logits, TOPK_FRAC)
            scores.append(torch.sigmoid(bag_logit).cpu().numpy())
    return np.concatenate(scores)


def case_activity_signs(rows):
    """Same per-case activity-level sign convention already established in
    this project for blob_signed/hp_prev_signed (median rgb_prev vs
    ACT_THRESH=2.7) -- reused here since TempFreq shows the exact same
    domain-dependent-sign-flip signature (low-activity domains LASIESTA/CDnet
    inverted, high-activity ShanghaiTech correct direction)."""
    by_case = bisfd.cw.rows_by_case(rows)
    return {c: (1.0 if float(np.median(bisfd.cw.matrix(r, ["rgb_prev"]).ravel())) > bisfd.ACT_THRESH else -1.0)
            for c, r in by_case.items()}


def run_once(rows, case_data, seed, sign_by_case):
    torch.manual_seed(seed)
    results = {}
    for held_out in DOMAINS:
        train_cases = {c: v for c, v in case_data.items() if v[2] != held_out}
        test_cases = {c: v for c, v in case_data.items() if v[2] == held_out}

        X_train, Y_train, _, _ = make_windows(train_cases, stride=STRIDE_TRAIN)
        X_test, Y_test, cases_test, _ = make_windows(test_cases, stride=STRIDE_TEST)
        print(f"  [seed={seed}] held_out={held_out} train={X_train.shape} test={X_test.shape}", flush=True)

        model = train_freq_model(X_train, Y_train)
        scores = predict_freq_model(model, X_test)
        auc = roc_auc_score(Y_test, scores) if len(set(Y_test.tolist())) > 1 else None

        signs = np.array([sign_by_case[c] for c in cases_test])
        signed_scores = np.where(signs > 0, scores, 1.0 - scores)
        auc_signed = roc_auc_score(Y_test, signed_scores) if len(set(Y_test.tolist())) > 1 else None

        print(f"  [seed={seed}] {held_out}: TempFreq AUC={auc:.4f}  signed AUC={auc_signed:.4f}", flush=True)
        results[held_out] = {"raw": auc, "signed": auc_signed}
    return results


def main():
    rows = bisfd.load_pool()
    rows = bisfd.add_signed(rows)
    sign_by_case = case_activity_signs(rows)
    n_neg = sum(1 for v in sign_by_case.values() if v < 0)
    print(f"activity sign: {n_neg}/{len(sign_by_case)} cases below ACT_THRESH={bisfd.ACT_THRESH} (sign=-1)", flush=True)

    print("extracting patch-signal time series for ALL cases...", flush=True)
    case_data = build_case_patch_signals(rows)

    SEEDS = [0, 1, 2]
    per_seed = []
    for seed in SEEDS:
        print(f"\n########## seed={seed} ##########", flush=True)
        per_seed.append(run_once(rows, case_data, seed, sign_by_case))

    print("\n=== Temporal-Frequency LODO results across seeds (raw vs sign-corrected) ===")
    for d in DOMAINS:
        raw_vals = [r[d]["raw"] for r in per_seed]
        signed_vals = [r[d]["signed"] for r in per_seed]
        print(f"  {d}: raw={np.mean(raw_vals):.4f}+-{np.std(raw_vals):.4f}  "
              f"signed={np.mean(signed_vals):.4f}+-{np.std(signed_vals):.4f}")

    for key in ["raw", "signed"]:
        macro = np.mean([np.mean([r[d][key] for d in DOMAINS]) for r in per_seed])
        worst = np.mean([min(r[d][key] for d in DOMAINS) for r in per_seed])
        print(f"\nmean macro LODO AUC across seeds ({key}): {macro:.4f}")
        print(f"mean worst-domain LODO AUC across seeds ({key}): {worst:.4f}")

    print("\n(reference) LightGBM macro=0.5841 worst=0.5496; current best 3-way fusion macro=0.6106 worst=0.5672")

    print("\nALL_DONE")


if __name__ == "__main__":
    main()
