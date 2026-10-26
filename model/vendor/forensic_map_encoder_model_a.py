"""
Domain-generalized forensic representation learning -- Model A only (per the
user's own ablation ordering: validate the plain encoder+classifier cleanly
before adding cross-domain contrastive loss (Model B) or domain-adversarial
training (Model C), which come in follow-up scripts once this is validated).

Unlike everything else built today (which reused the pre-extracted 17 scalar
features -- mean/std of each forensic map, collapsing spatial structure),
this operates on the actual 2D forensic maps per frame:
  - gray temporal residual  |gray_t - gray_{t-1}|
  - high-pass residual      img - GaussianBlur(img, 7)   (fine detail)
  - Laplacian magnitude     |Laplacian(img)|
  - "texture" residual      img - GaussianBlur(img, 15)  (coarser structure)
(reusing this project's existing gray01/blur/highpass/lap_abs definitions
from blind_inpaint_detector_prev_warning_border.py for consistency), each
resized to 32x32. A small per-frame 2D CNN embeds each map-stack into a
64-dim vector; a 1D-TCN aggregates a W=16 frame window of these embeddings
into a trajectory representation; a single linear head classifies
real/forged. Still fully blind: no GT mask, no YOLO, no object position --
only these 4 pixel-derived residual maps.

Evaluation: Leave-One-Domain-Out (train on 2 of LASIESTA/ShanghaiTech/CDnet,
test on the entirely-unseen 3rd), compared against the LightGBM base_context
baseline under the identical LODO split.
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

# cudnn's default (non-deterministic, benchmark-mode) conv algorithms made
# train_model_a's AUC vary by ~0.002-0.004 across runs with the SAME
# torch.manual_seed(seed) -- confirmed 2026-08-10 via a same-process double-run.
# This alone was comparable in size to several past ADOPT/REJECT deltas.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import train_eval_bisfd_pooled_split as bisfd

# 2026-09-22: extract_maps_for_video now decodes one extra leading frame and
# drops it, so array index 0 aligns with CSV frame=1 instead of frame=0 (see
# that function's docstring). Sentinel for the collaborator's own opt-in
# alignment overlay to detect this fix is already applied here and skip
# re-applying its own -- otherwise the two corrections stack and shift
# labels one frame the other way.
FRAME_ALIGN_FIXED = True

ALL_DATASETS_FINAL = "/home/tako/songhye/project4/all_datasets_final"
SHANGHAITECH_TESTSET = "/home/tako/songhye/project4/dataset_add/shanghaitech_testset"
STREETSCENE_TESTSET = "/home/tako/songhye/project4/dataset_add/streetscene_testset"

MAP_SIZE = 32
W = 16
STRIDE_TRAIN = 4
STRIDE_TEST = 2
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DOMAINS = ["lasiesta", "shanghaitech", "cdnet", "streetscene"]


# =========================================================
# Case -> video file resolution
# =========================================================
def resolve_video_path(case, dataset):
    if dataset == "lasiesta":
        if case.startswith("PRISTINE_LAS_") or case.startswith("PRISTINE_O_"):
            dirname = case[len("PRISTINE_"):]
            return f"{ALL_DATASETS_FINAL}/lasiesta/{dirname}/pristine.mp4"
        return f"{ALL_DATASETS_FINAL}/lasiesta/{case}/forged.mp4"
    if dataset == "shanghaitech":
        scene, variant = case.rsplit("_", 1)
        fname = {"pristine": "pristine.mp4", "propainter": "propainter_forged.mp4",
                  "minimax": "minimax_forged.mp4"}[variant]
        return f"{SHANGHAITECH_TESTSET}/{scene}/{fname}"
    if dataset == "streetscene":
        scene, variant = case.rsplit("_", 1)
        fname = {"pristine": "pristine.mp4", "propainter": "propainter_forged.mp4",
                  "minimax": "minimax_forged.mp4"}[variant]
        return f"{STREETSCENE_TESTSET}/{scene}/{fname}"
    if dataset == "cdnet":
        m = re.match(r"^(.*)_(propainter|minimax|PRISTINE)$", case)
        window, variant = m.group(1), m.group(2)
        fname = {"PRISTINE": "pristine.mp4", "propainter": "forged.mp4", "minimax": "minimax_forged.mp4"}[variant]
        for subdir in ["cdnet", "cdnet_heldout"]:
            path = f"{ALL_DATASETS_FINAL}/{subdir}/{window}/{fname}"
            if os.path.exists(path):
                return path
        raise FileNotFoundError(case)
    raise ValueError(dataset)


# =========================================================
# Map extraction (reusing this project's blur/highpass/lap_abs definitions)
# =========================================================
def blur(img, k):
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), 0)


def extract_maps_for_video(path, n_frames_needed):
    """Decodes n_frames_needed+1 raw frames and drops the first result before
    returning -- the CSV's "frame" column comes from blind_inpaint_detector_
    prev_warning_border.py's `for t in range(1, n)` loop, so CSV frame=f
    describes evidence computed AT the (0-indexed) video frame f, using frame
    f-1 as the diff reference. Reading sequentially from frame 0 the way this
    function always has, maps[k] (before this fix) described video frame k --
    off by one from what CSV frame=k actually means once callers do
    `idx = frame - 1`. Dropping the artificial first (zero-diff, prev_gray is
    None) entry re-aligns maps[0] with video frame 1 = CSV frame=1, so
    `idx = frame - 1` and make_windows' `frames.append(t + 1)` both stay
    correct with no other caller-side change. Also removes the zero-filled
    first-frame diff from each case's per-video robust normalization stats.
    2026-09-22, cross-checked against the friend's independent mask-alignment
    re-verification (99.44% -> 100% match on the extended clips)."""
    cap = cv2.VideoCapture(path)
    maps = []
    prev_gray = None
    count = 0
    while count < n_frames_needed + 1:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if prev_gray is None:
            gray_diff = np.zeros_like(gray)
        else:
            gray_diff = np.abs(gray - prev_gray)
        hp = np.abs(gray - blur(gray, 7))
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        texture = np.abs(gray - blur(gray, 15))

        stack = np.stack([
            cv2.resize(gray_diff, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_AREA),
            cv2.resize(hp, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_AREA),
            cv2.resize(lap, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_AREA),
            cv2.resize(texture, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_AREA),
        ], axis=0).astype(np.float32)  # (4, 32, 32)

        maps.append(stack)
        prev_gray = gray
        count += 1
    cap.release()
    maps = maps[1:]
    return np.stack(maps, axis=0) if maps else np.zeros((0, 4, MAP_SIZE, MAP_SIZE), dtype=np.float32)


def build_case_maps(rows):
    """One entry per case: (maps[n,4,32,32], labels[n], domain, case_name,
    raw_signal_strength). raw_signal_strength is the mean abs magnitude of the
    highpass+laplacian channels BEFORE per-case normalization -- a cheap
    sharpness/detail proxy kept around for evidence-reliability gating
    (normalization erases this scale, so it must be captured here)."""
    by_case = bisfd.cw.rows_by_case(rows)
    case_data = {}
    t0 = time.time()
    n_done = 0
    for case, case_rows in by_case.items():
        case_rows.sort(key=lambda r: r["frame"])
        dataset = case_rows[0]["dataset"]
        n_needed = case_rows[-1]["frame"]  # frames are 1-indexed in the CSV
        video_path = resolve_video_path(case, dataset)
        maps = extract_maps_for_video(video_path, n_needed)
        raw_signal_strength = float(np.mean(np.abs(maps[:, 1:3]))) if maps.shape[0] > 0 else 0.0
        # per-video robust normalization of each of the 4 map channels (self-reference,
        # matches this project's established convention)
        if maps.shape[0] > 0:
            med = np.median(maps, axis=(0, 2, 3), keepdims=True)
            mad = np.median(np.abs(maps - med), axis=(0, 2, 3), keepdims=True) + 1e-3
            maps = (maps - med) / mad
        labels = np.zeros(maps.shape[0], dtype=np.float32)
        for r in case_rows:
            idx = r["frame"] - 1
            if 0 <= idx < len(labels):
                labels[idx] = bisfd.cw.safe_float(r.get("gt_label", 0.0))
        case_data[case] = (maps, labels, dataset, raw_signal_strength)
        n_done += 1
        if n_done % 20 == 0:
            print(f"    extracted maps for {n_done}/{len(by_case)} cases ({time.time()-t0:.1f}s)", flush=True)
    print(f"  map extraction done: {n_done} cases in {time.time()-t0:.1f}s", flush=True)
    return case_data


def make_windows(case_data, stride):
    """frames returned are 1-indexed (t+1) to match the CSV's 'frame' column,
    so windows can be aligned back to specific (case, frame) rows -- needed
    whenever CNN scores must be fused/compared sample-for-sample against
    another model's per-row predictions (see lodo_fusion.py)."""
    windows, ys, doms, cases, frames = [], [], [], [], []
    half = W // 2
    for case, (maps, labels, dataset, _raw_signal) in case_data.items():
        n = maps.shape[0]
        if n == 0:
            continue
        padded = np.pad(maps, ((half, half), (0, 0), (0, 0), (0, 0)), mode="edge")  # (n+2half,4,32,32)
        for t in range(0, n, stride):
            seg = padded[t: t + W]  # (W,4,32,32)
            windows.append(seg.astype(np.float32))
            ys.append(labels[t])
            doms.append(dataset)
            cases.append(case)
            frames.append(t + 1)
    X = np.stack(windows, axis=0)  # (N, W, 4, 32, 32)
    Y = np.asarray(ys, dtype=np.float32)
    return X, Y, doms, cases, frames


# =========================================================
# Model A: frame CNN + 1D-TCN + classification head
# =========================================================
class FrameCNN(nn.Module):
    def __init__(self, in_ch=4, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # -> (32,1,1)
        )
        self.proj = nn.Linear(32, emb_dim)

    def forward(self, x):
        # x: (B, in_ch, 32, 32)
        h = self.net(x).flatten(1)
        return self.proj(h)


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ModelA(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.frame_cnn = FrameCNN(in_ch=4, emb_dim=emb_dim)
        self.tcn = nn.Sequential(
            TCNBlock(emb_dim, emb_dim, dilation=1),
            TCNBlock(emb_dim, emb_dim, dilation=2),
            TCNBlock(emb_dim, emb_dim, dilation=4),
        )
        self.cls_head = nn.Linear(emb_dim, 1)

    def forward(self, x):
        # x: (B, W, 4, 32, 32)
        B, Wt, C, H, Wd = x.shape
        x = x.view(B * Wt, C, H, Wd)
        emb = self.frame_cnn(x)             # (B*W, emb_dim)
        emb = emb.view(B, Wt, -1).transpose(1, 2)  # (B, emb_dim, W)
        h = self.tcn(emb)                    # (B, emb_dim, W)
        z = h.mean(dim=2)                    # (B, emb_dim)
        return self.cls_head(z).squeeze(-1)


def train_model_a(X_train, Y_train, epochs=EPOCHS):
    model = ModelA().to(DEVICE)
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
            xb = Xt[idx].to(DEVICE)
            yb = Yt[idx].to(DEVICE)
            logit = model(xb)
            loss = bce(logit, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        print(f"    epoch {epoch+1}/{epochs} loss={total_loss/n:.4f}", flush=True)
    return model


def predict_model_a(model, X_test, batch_size=256):
    model.eval()
    Xt = torch.from_numpy(X_test)
    scores = []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], batch_size):
            xb = Xt[i:i + batch_size].to(DEVICE)
            logit = model(xb)
            scores.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(scores)


# =========================================================
# LightGBM LODO baseline
# =========================================================
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


FEATS = bisfd.cw.BASE_FEATURES + ["blob_score", "blob_signed", "hp_prev_signed"]


def main():
    rows = bisfd.load_pool()
    rows = bisfd.add_signed(rows)

    print("extracting forensic maps for ALL cases (one-time cost, cached in memory)...", flush=True)
    case_data = build_case_maps(rows)

    results = {}
    for held_out in DOMAINS:
        print(f"\n=== LODO: held-out domain = {held_out} ===", flush=True)
        train_cases = {c: v for c, v in case_data.items() if v[2] != held_out}
        test_cases = {c: v for c, v in case_data.items() if v[2] == held_out}

        X_train, Y_train, _, _, _ = make_windows(train_cases, stride=STRIDE_TRAIN)
        X_test, Y_test, _, _, _ = make_windows(test_cases, stride=STRIDE_TEST)
        print(f"  train windows={X_train.shape} test windows={X_test.shape}", flush=True)

        print("  training Model A...", flush=True)
        model = train_model_a(X_train, Y_train)
        scores = predict_model_a(model, X_test)
        auc_encoder = roc_auc_score(Y_test, scores) if len(set(Y_test.tolist())) > 1 else None
        print(f"  Model A AUC ({held_out} held out): {auc_encoder}", flush=True)

        train_rows = [r for r in rows if r["dataset"] != held_out]
        test_rows = [r for r in rows if r["dataset"] == held_out]
        auc_baseline = lgbm_lodo_baseline(train_rows, test_rows, FEATS)
        print(f"  LightGBM baseline AUC ({held_out} held out): {auc_baseline}", flush=True)

        results[held_out] = {"model_a": auc_encoder, "baseline": auc_baseline}

    print("\n=== LODO comparison summary ===")
    print(f"{'held-out domain':<16} {'LightGBM baseline':>20} {'Model A (CNN+TCN)':>20}")
    for d in DOMAINS:
        print(f"{d:<16} {results[d]['baseline']:.4f}{'':>15} {results[d]['model_a']:.4f}")
    macro_b = np.mean([results[d]["baseline"] for d in DOMAINS])
    macro_a = np.mean([results[d]["model_a"] for d in DOMAINS])
    print(f"\nmacro LODO AUC: baseline={macro_b:.4f} model_a={macro_a:.4f}")
    print("\nALL_DONE")


if __name__ == "__main__":
    main()
