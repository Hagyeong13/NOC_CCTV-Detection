import os
import cv2
import csv
import json
import argparse
import glob
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


EPS = 1e-8


# =========================================================
# Utility
# =========================================================

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def safe_float(x):
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return 0.0
        return x
    except Exception:
        return 0.0


def robust_mad(x):
    x = np.asarray(x, dtype=np.float32)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + EPS
    return med, mad


def robust_norm(x, cap=None):
    x = np.asarray(x, dtype=np.float32)
    med, mad = robust_mad(x)
    z = (x - med) / mad
    if cap is not None:
        z = np.clip(z, -cap, cap)
    return z.astype(np.float32)


def positive_robust_map(m, cap=6.0):
    z = robust_norm(m.reshape(-1), cap=cap).reshape(m.shape)
    return np.maximum(z, 0).astype(np.float32)


def robust_z(x, ref):
    ref = np.asarray(ref, dtype=np.float32)
    if ref.size == 0:
        return 0.0
    med = np.median(ref)
    mad = np.median(np.abs(ref - med)) + EPS
    return float((x - med) / mad)


def minmax01(x):
    x = np.asarray(x, dtype=np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < EPS:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def moving_average(x, win=13):
    x = np.asarray(x, dtype=np.float32)
    if win <= 1 or len(x) < win:
        return x.copy()

    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.zeros_like(x, dtype=np.float32)

    for i in range(len(x)):
        out[i] = np.mean(xp[i:i + win])

    return out


def topk_mean(values, ratio=0.10):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0

    k = max(1, int(values.size * ratio))
    return float(np.mean(np.sort(values)[-k:]))


def norm_series01(x, p_low=2, p_high=98):
    x = np.asarray(x, dtype=np.float32)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)

    if hi - lo < EPS:
        return np.zeros_like(x, dtype=np.float32)

    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


# =========================================================
# Video I/O
# =========================================================

def read_video(path, resize_width=None, max_frames=None):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1:
        fps = 30.0

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if resize_width is not None:
            h, w = frame.shape[:2]
            scale = resize_width / float(w)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)

        frames.append(frame)

        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    if not frames:
        raise ValueError(f"No frames loaded from {path}")

    return frames, fps


def write_video(path, frames, fps):
    if not frames:
        return

    ensure_dir(os.path.dirname(path))

    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    for f in frames:
        out.write(np.clip(f, 0, 255).astype(np.uint8))

    out.release()


def gray01(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def bgr01(frame):
    return frame.astype(np.float32) / 255.0


# =========================================================
# Image features
# =========================================================

def blur(img, k=7):
    k = int(k)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), 0)


def highpass(img):
    return img.astype(np.float32) - blur(img, 7)


def lap_abs(img):
    return np.abs(cv2.Laplacian(img.astype(np.float32), cv2.CV_32F, ksize=3))


def sobel_mag(img):
    gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def local_mean(img, k=65):
    k = int(k)
    if k % 2 == 0:
        k += 1
    k = max(3, k)
    return cv2.GaussianBlur(img.astype(np.float32), (k, k), 0)


def local_low(feature, k=65):
    """
    주변 local 평균보다 현재 feature가 낮은 정도.
    texture / edge / high-frequency loss를 보기 위한 값.
    """
    return np.maximum(0, local_mean(feature, k) - feature)


def dct_hf_ratio(patch):
    p = patch.astype(np.float32)

    if p.shape[0] < 8 or p.shape[1] < 8:
        return 0.0

    p = cv2.resize(p, (64, 64), interpolation=cv2.INTER_AREA)
    d = cv2.dct(p)
    e = d * d

    h, w = e.shape
    yy, xx = np.mgrid[0:h, 0:w]
    rr = (xx + yy) / float(h + w)

    hf = np.sum(e[rr > 0.42])
    total = np.sum(e) + EPS

    return float(hf / total)


# =========================================================
# Alignment
# =========================================================

def estimate_affine(src_gray, dst_gray):
    """
    src_gray를 dst_gray에 맞추기 위한 affine transform 추정.
    실패하면 identity 반환.
    """
    src8 = np.clip(src_gray * 255, 0, 255).astype(np.uint8)
    dst8 = np.clip(dst_gray * 255, 0, 255).astype(np.uint8)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        50,
        1e-5
    )

    try:
        cv2.findTransformECC(
            dst8,
            src8,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5
        )
        return warp
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32)


def apply_affine(img, warp, out_shape):
    h, w = out_shape[:2]

    return cv2.warpAffine(
        img.astype(np.float32),
        warp,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT
    )


# =========================================================
# Patch helpers
# =========================================================

def make_patches(h, w, patch, stride):
    patches = []

    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            patches.append((x, y, patch, patch))

    return patches


def crop(img, p):
    x, y, w, h = p
    return img[y:y + h, x:x + w]


def patch_mean(img, p):
    return float(np.mean(crop(img, p)))


def neighbor_indices(idx, grid_w, grid_h):
    y = idx // grid_w
    x = idx % grid_w

    ids = []

    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue

            nx = x + dx
            ny = y + dy

            if 0 <= nx < grid_w and 0 <= ny < grid_h:
                ids.append(ny * grid_w + nx)

    return ids


# =========================================================
# Warning overlay / Save / Plot
# =========================================================

def make_warning_overlay(frame, text="WARNING", use_english=True):
    """
    의심 프레임에서 영상 전체 외곽에 빨간 테두리를 두르고,
    상단 좌측에 작은 WARNING 라벨만 표시.
    위치 박스/heatmap/큰 배너는 표시하지 않음.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # 전체 빨간 테두리
    border_thick = max(6, int(min(h, w) * 0.018))

    cv2.rectangle(
        out,
        (border_thick // 2, border_thick // 2),
        (w - border_thick // 2, h - border_thick // 2),
        (0, 0, 255),
        border_thick
    )

    # 작은 WARNING 라벨
    label = "WARNING" if use_english else text

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, w / 1400.0)
    thickness = max(1, int(w / 900))

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    pad_x = max(10, int(w * 0.012))
    pad_y = max(6, int(h * 0.008))

    box_w = tw + pad_x * 2
    box_h = th + baseline + pad_y * 2

    x1 = max(border_thick + 8, int(w * 0.035))
    y1 = max(border_thick + 8, int(h * 0.035))
    x2 = min(w - border_thick - 8, x1 + box_w)
    y2 = min(h - border_thick - 8, y1 + box_h)

    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 130), 2)

    tx = x1 + pad_x
    ty = y1 + pad_y + th

    cv2.putText(
        out,
        label,
        (tx, ty),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return out


def save_json(path, obj):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(path, rows):
    if not rows:
        return

    ensure_dir(os.path.dirname(path))

    keys = []

    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def shade_segments(ax, result):
    for seg in result.get("suspicious_segments", []):
        ax.axvspan(
            seg["start_time_sec"],
            seg["end_time_sec"],
            alpha=0.18,
            color="tab:red",
            label="detected segment"
        )

    gt = result.get("gt_interval")

    if gt:
        ax.axvspan(
            gt[0],
            gt[1],
            alpha=0.12,
            color="green",
            label="GT interval"
        )


def save_score_plot(path, result, threshold):
    if not HAS_MPL:
        return

    rows = result["frame_scores"]
    fps = float(result["fps"])

    t = np.array([r["frame"] / fps for r in rows], dtype=np.float32)

    final = np.array([r["smooth_score"] for r in rows], dtype=np.float32)
    frame_score = np.array([r["score"] for r in rows], dtype=np.float32)
    transition = np.array([r.get("transition", 0.0) for r in rows], dtype=np.float32)
    stable = np.array([r.get("stable_score", 0.0) for r in rows], dtype=np.float32)
    texture_loss = np.array([r.get("texture_loss", 0.0) for r in rows], dtype=np.float32)

    ensure_dir(os.path.dirname(path))

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        t,
        final,
        linewidth=2.6,
        color="tab:blue",
        label="SMOOTHED FINAL"
    )
    ax.plot(t, frame_score, linestyle="--", linewidth=1.1, alpha=0.78, label="frame score")
    ax.plot(t, transition, linestyle="--", linewidth=1.2, alpha=0.78, label="transition")
    ax.plot(t, stable, linestyle="--", linewidth=1.2, alpha=0.78, label="stable")
    ax.plot(t, texture_loss, linestyle="--", linewidth=1.2, alpha=0.78, label="texture loss")
    ax.axhline(threshold, linestyle="--", linewidth=1.4, label=f"threshold={threshold}")

    shade_segments(ax, result)

    ax.set_title("Score Timeline - SMOOTHED FINAL Focus")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_feature_plot(path, result):
    if not HAS_MPL:
        return

    rows = result["frame_scores"]
    fps = float(result["fps"])

    t = np.array([r["frame"] / fps for r in rows], dtype=np.float32)

    keys = [
        "rgb_prev",
        "gray_prev",
        "hp_prev",
        "edge_prev",
        "lap_prev",
        "stable",
        "hp_low",
        "lap_low",
        "dct_low",
    ]

    ensure_dir(os.path.dirname(path))

    fig, ax = plt.subplots(figsize=(14, 6))

    for k in keys:
        if k not in rows[0]:
            continue

        y = np.array([r.get(k, 0.0) for r in rows], dtype=np.float32)
        y = norm_series01(y)

        ax.plot(t, y, linewidth=1.25, label=k)

    shade_segments(ax, result)

    ax.set_title("Prev-only Blind Feature Timeline Normalized")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Normalized feature")
    ax.grid(True, alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), ncol=2)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_top_events(path, result, top_n=20):
    rows = result["frame_scores"]
    fps = float(result["fps"])

    top = sorted(
        rows,
        key=lambda r: r.get("smooth_score", 0.0),
        reverse=True
    )[:top_n]

    events = []

    for rank, r in enumerate(top, 1):
        frame = int(r["frame"])

        item = {
            "rank": rank,
            "frame": frame,
            "time_sec": safe_float(frame / fps),
            "score": safe_float(r.get("smooth_score", 0.0)),
            "suspicious": bool(r.get("suspicious", False)),
        }

        for k, v in r.items():
            if k not in item and isinstance(v, (int, float, np.float32, np.float64)):
                item[k] = safe_float(v)

        events.append(item)

    save_json(path, events)


def print_top_events(result, top_n=10):
    rows = result["frame_scores"]
    fps = float(result["fps"])

    top = sorted(
        rows,
        key=lambda r: r.get("smooth_score", 0.0),
        reverse=True
    )[:top_n]

    keys = [
        "transition",
        "stable_score",
        "texture_loss",
        "rgb_prev",
        "hp_prev",
        "edge_prev",
        "dct_low",
    ]

    print("\n[TOP EVENTS]")

    for rank, r in enumerate(top, 1):
        frame = int(r["frame"])
        sec = frame / fps
        score = r.get("smooth_score", 0.0)

        strong = sorted(
            [k for k in keys if k in r],
            key=lambda k: r.get(k, 0.0),
            reverse=True
        )[:4]

        msg = f"#{rank:02d} frame={frame}, time={sec:.2f}s, score={score:.3f}"

        if strong:
            msg += " | strong=" + ", ".join([f"{k}:{r.get(k, 0.0):.2f}" for k in strong])

        print(msg)


def safe_video_name(path):
    base = os.path.splitext(os.path.basename(path))[0]

    if base.lower() in ("propainter_out", "rpopainter_out"):
        parent = os.path.basename(os.path.dirname(path))
        if parent:
            base = parent

    safe = []

    for ch in base:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")

    name = "".join(safe).strip("_")
    return name or "video"


def unique_output_dir(root, video_path, used_names):
    name = safe_video_name(video_path)
    candidate = name
    idx = 2

    while candidate in used_names:
        candidate = f"{name}_{idx}"
        idx += 1

    used_names.add(candidate)
    return os.path.join(root, candidate)


def expand_video_inputs(args):
    videos = []

    if args.video:
        videos.append(args.video)

    for path in args.videos:
        videos.append(path)

    for pattern in args.video_glob:
        videos.extend(sorted(glob.glob(pattern)))

    for video_dir in args.video_dir:
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm", "*.m4v"):
            videos.extend(sorted(glob.glob(os.path.join(video_dir, ext))))

    for dataset_dir in args.dataset_dir:
        patterns = [
            os.path.join(dataset_dir, "*", "propainter_out.mp4"),
            os.path.join(dataset_dir, "*", "rpopainter_out.mp4"),
        ]

        for pattern in patterns:
            videos.extend(sorted(glob.glob(pattern)))

    seen = set()
    unique = []

    for path in videos:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(path)

    return unique


def auto_original_for(video_path):
    candidate = os.path.join(os.path.dirname(video_path), "original.mp4")
    if os.path.exists(candidate):
        return candidate
    return None


def auto_meta_for(video_path):
    candidate = os.path.join(os.path.dirname(video_path), "meta.json")
    if os.path.exists(candidate):
        return candidate
    return None


def resize_like(frame, shape):
    h, w = shape[:2]
    if frame.shape[:2] == (h, w):
        return frame
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def compare_with_original(original_path, detected_frames, frame_scores, args):
    original_frames, original_fps = read_video(
        original_path,
        resize_width=args.resize_width,
        max_frames=args.max_frames
    )

    n = min(len(original_frames), len(detected_frames))
    rows = []
    changed_total = 0
    detected_total = 0
    detected_changed = 0

    for i in range(n):
        original = resize_like(original_frames[i], detected_frames[i].shape)
        detected = detected_frames[i]

        diff = cv2.absdiff(original, detected)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        mean_abs = float(np.mean(gray_diff))
        changed_pixel_ratio = float(np.mean(gray_diff >= args.compare_pixel_threshold))
        changed = (
            mean_abs >= args.compare_mean_threshold
            or changed_pixel_ratio >= args.compare_ratio_threshold
        )

        rows.append({
            "frame": i,
            "time_sec": safe_float(i / float(original_fps)),
            "mean_abs_diff": safe_float(mean_abs),
            "changed_pixel_ratio": safe_float(changed_pixel_ratio),
            "changed_vs_original": bool(changed),
        })

        if changed:
            changed_total += 1

    suspicious_by_frame = {
        int(r["frame"]): bool(r.get("suspicious", False))
        for r in frame_scores
    }

    for r in rows:
        suspicious = suspicious_by_frame.get(int(r["frame"]), False)
        r["suspicious"] = bool(suspicious)

        if suspicious:
            detected_total += 1

        if r["changed_vs_original"] and suspicious:
            detected_changed += 1

    false_positive = detected_total - detected_changed
    missed_changed = changed_total - detected_changed

    return {
        "original_path": original_path,
        "compared_frames": n,
        "compare_pixel_threshold": args.compare_pixel_threshold,
        "compare_ratio_threshold": args.compare_ratio_threshold,
        "compare_mean_threshold": args.compare_mean_threshold,
        "changed_frames": changed_total,
        "detected_suspicious_frames": detected_total,
        "detected_changed_frames": detected_changed,
        "missed_changed_frames": missed_changed,
        "false_positive_frames": false_positive,
        "changed_frame_percent": safe_float(100.0 * changed_total / max(n, 1)),
        "detected_changed_percent": safe_float(100.0 * detected_changed / max(changed_total, 1)),
        "precision_percent": safe_float(100.0 * detected_changed / max(detected_total, 1)),
        "frame_rows": rows,
    }


def interval_frames(start_frame, end_frame, total_frames):
    start = max(0, int(start_frame))
    end = min(total_frames - 1, int(end_frame))

    if end < start:
        return set()

    return set(range(start, end + 1))


def compare_with_gt_timeline(meta_path, result):
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    total_frames = int(result.get("num_frames", 0))
    fps = float(result.get("fps", 30.0))
    gt_start = int(meta.get("forge_start", -1))
    gt_end = int(meta.get("forge_end", -1))

    frame_ids = {
        int(row.get("frame", idx))
        for idx, row in enumerate(result.get("frame_scores", []))
    }
    if not frame_ids:
        frame_ids = set(range(total_frames))

    if isinstance(meta.get("gt_frames"), list):
        gt_frames = {int(frame) for frame in meta["gt_frames"] if int(frame) in frame_ids}
        if gt_frames:
            gt_start = min(gt_frames)
            gt_end = max(gt_frames)
        else:
            gt_frames = interval_frames(gt_start, gt_end, total_frames)
    else:
        gt_frames = interval_frames(gt_start, gt_end, total_frames)
    detected_frames = set()

    for seg in result.get("suspicious_segments", []):
        detected_frames.update(
            interval_frames(seg["start_frame"], seg["end_frame"], total_frames)
        )

    detected_frames = detected_frames & frame_ids
    gt_frames = gt_frames & frame_ids
    all_frames = frame_ids
    tp = len(gt_frames & detected_frames)
    fn = len(gt_frames - detected_frames)
    fp = len(detected_frames - gt_frames)
    tn = len(all_frames - gt_frames - detected_frames)

    def pct(x):
        return safe_float(100.0 * x / max(total_frames, 1))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)

    return {
        "meta_path": meta_path,
        "gt_start_frame": gt_start,
        "gt_end_frame": gt_end,
        "total_frames": total_frames,
        "total_time_sec": safe_float(total_frames / fps),
        "gt_frames": len(gt_frames),
        "detected_frames": len(detected_frames),
        "matched_manipulated_frames": tp,
        "missed_manipulated_frames": fn,
        "false_alarm_frames": fp,
        "matched_clean_frames": tn,
        "gt_time_sec": safe_float(len(gt_frames) / fps),
        "detected_time_sec": safe_float(len(detected_frames) / fps),
        "matched_manipulated_time_sec": safe_float(tp / fps),
        "missed_manipulated_time_sec": safe_float(fn / fps),
        "false_alarm_time_sec": safe_float(fp / fps),
        "gt_percent_of_full_video": pct(len(gt_frames)),
        "detected_percent_of_full_video": pct(len(detected_frames)),
        "matched_manipulated_percent_of_full_video": pct(tp),
        "missed_manipulated_percent_of_full_video": pct(fn),
        "false_alarm_percent_of_full_video": pct(fp),
        "clean_matched_percent_of_full_video": pct(tn),
        "timeline_accuracy_percent": pct(tp + tn),
        "gt_precision_percent": safe_float(100.0 * precision),
        "gt_recall_percent": safe_float(100.0 * recall),
        "gt_f1_percent": safe_float(100.0 * f1),
    }


def process_video(video_path, out_dir, args):
    ensure_dir(out_dir)

    frames, fps = read_video(
        video_path,
        resize_width=args.resize_width,
        max_frames=args.max_frames
    )

    result, overlay_frames = blind_detect_prev_warning(frames, fps, args)
    result["video_path"] = video_path
    result["settings"] = vars(args).copy()
    result["settings"]["current_video"] = video_path

    original_path = args.compare_original or auto_original_for(video_path)

    if original_path:
        comparison = compare_with_original(
            original_path,
            frames,
            result["frame_scores"],
            args
        )
        result["original_comparison"] = {
            k: v for k, v in comparison.items() if k != "frame_rows"
        }

    meta_path = auto_meta_for(video_path)

    if meta_path:
        result["gt_timeline_comparison"] = compare_with_gt_timeline(meta_path, result)

    out_json = os.path.join(out_dir, "result.json")
    out_csv = os.path.join(out_dir, "frame_scores.csv")
    out_overlay = os.path.join(out_dir, "overlay_warning.mp4")
    out_score = os.path.join(out_dir, "score_timeline.png")
    out_feat = os.path.join(out_dir, "feature_timeline.png")
    out_top = os.path.join(out_dir, "top_events.json")
    out_compare_csv = os.path.join(out_dir, "original_comparison.csv")

    save_json(out_json, result)
    save_csv(out_csv, result["frame_scores"])
    write_video(out_overlay, overlay_frames, fps)
    save_score_plot(out_score, result, args.threshold)
    save_feature_plot(out_feat, result)
    save_top_events(out_top, result, 20)

    if original_path:
        save_csv(out_compare_csv, comparison["frame_rows"])

    print("\n[DONE]")
    print("video:", video_path)
    print("mode:", result["mode"])
    print("json:", out_json)
    print("csv:", out_csv)
    print("overlay warning video:", out_overlay)
    print("score plot:", out_score)
    print("feature plot:", out_feat)
    print("top events:", out_top)
    if original_path:
        print("original comparison csv:", out_compare_csv)
        print(
            "detected changed:",
            f"{comparison['detected_changed_percent']:.2f}% "
            f"({comparison['detected_changed_frames']}/{comparison['changed_frames']} changed frames)"
        )
    if meta_path:
        gt_comp = result["gt_timeline_comparison"]
        print(
            "full timeline match:",
            f"{gt_comp['timeline_accuracy_percent']:.2f}% | "
            f"GT {gt_comp['gt_percent_of_full_video']:.2f}%, "
            f"detected {gt_comp['detected_percent_of_full_video']:.2f}%, "
            f"missed {gt_comp['missed_manipulated_percent_of_full_video']:.2f}%, "
            f"false alarm {gt_comp['false_alarm_percent_of_full_video']:.2f}%"
        )
    print("segments:", result["suspicious_segments"])

    print_top_events(result, 10)

    summary = {
        "video": video_path,
        "out_dir": out_dir,
        "result_json": out_json,
        "score_plot": out_score,
        "feature_plot": out_feat,
        "overlay_video": out_overlay,
        "top_events": out_top,
        "segments": result["suspicious_segments"],
    }

    if original_path:
        summary["original_comparison"] = result["original_comparison"]

    if meta_path:
        summary["gt_timeline_comparison"] = result["gt_timeline_comparison"]

    return summary


def flatten_batch_summary(summaries):
    rows = []

    for item in summaries:
        row = {
            "video": item.get("video", ""),
            "out_dir": item.get("out_dir", ""),
            "result_json": item.get("result_json", ""),
            "score_plot": item.get("score_plot", ""),
            "overlay_video": item.get("overlay_video", ""),
            "segments_count": len(item.get("segments", [])),
        }

        comparison = item.get("original_comparison", {})
        for key in [
            "original_path",
            "compared_frames",
            "changed_frames",
            "detected_suspicious_frames",
            "detected_changed_frames",
            "missed_changed_frames",
            "false_positive_frames",
            "changed_frame_percent",
            "detected_changed_percent",
            "precision_percent",
        ]:
            row[key] = comparison.get(key, "")

        gt_comparison = item.get("gt_timeline_comparison", {})
        for key in [
            "gt_start_frame",
            "gt_end_frame",
            "total_frames",
            "total_time_sec",
            "gt_frames",
            "detected_frames",
            "matched_manipulated_frames",
            "missed_manipulated_frames",
            "false_alarm_frames",
            "matched_clean_frames",
            "gt_time_sec",
            "detected_time_sec",
            "matched_manipulated_time_sec",
            "missed_manipulated_time_sec",
            "false_alarm_time_sec",
            "gt_percent_of_full_video",
            "detected_percent_of_full_video",
            "matched_manipulated_percent_of_full_video",
            "missed_manipulated_percent_of_full_video",
            "false_alarm_percent_of_full_video",
            "clean_matched_percent_of_full_video",
            "timeline_accuracy_percent",
            "gt_precision_percent",
            "gt_recall_percent",
            "gt_f1_percent",
        ]:
            row[key] = gt_comparison.get(key, "")

        rows.append(row)

    return rows


# =========================================================
# Detector
# =========================================================

def group_segments(rows, threshold, min_len=5, bridge_gap=20, low_threshold=None):
    scores = [r.get("smooth_score", 0.0) for r in rows]

    if low_threshold is None:
        flags = [s >= threshold for s in scores]
    else:
        flags = [False] * len(scores)

        for seed_idx, score in enumerate(scores):
            if score < threshold:
                continue

            left = seed_idx
            while left > 0 and scores[left - 1] >= low_threshold:
                left -= 1

            right = seed_idx
            while right + 1 < len(scores) and scores[right + 1] >= low_threshold:
                right += 1

            for i in range(left, right + 1):
                flags[i] = True

    segs = []
    start = None

    for i, f in enumerate(flags):
        if f and start is None:
            start = i

        elif not f and start is not None:
            end = i - 1

            if end - start + 1 >= min_len:
                segs.append([start, end])

            start = None

    if start is not None:
        end = len(flags) - 1

        if end - start + 1 >= min_len:
            segs.append([start, end])

    if not segs:
        return []

    merged = [segs[0]]

    for s, e in segs[1:]:
        if s - merged[-1][1] <= bridge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    return merged


def local_mean_score(rows, start, end, key):
    vals = [rows[i].get(key, 0.0) for i in range(start, end + 1)]
    if not vals:
        return 0.0
    return float(np.mean(vals))


def trim_segment_tails(rows, segs, window=9, min_len=5, score_threshold=0.75):
    if not segs:
        return []

    trimmed = []
    window = max(1, int(window))

    for start, end in segs:
        left = int(start)
        right = int(end)

        while left + window - 1 <= right:
            local_end = min(right, left + window - 1)
            if local_mean_score(rows, left, local_end, "smooth_score") >= score_threshold:
                break
            left += 1

        while right - window + 1 >= left:
            local_start = max(left, right - window + 1)
            if local_mean_score(rows, local_start, right, "smooth_score") >= score_threshold:
                break
            right -= 1

        if right - left + 1 >= min_len:
            trimmed.append([left, right])

    return trimmed


def segment_length(segs):
    return sum(max(0, int(e) - int(s) + 1) for s, e in segs)


def temporal_segment_score(rows, args):
    smooth = np.array([r.get("smooth_score", 0.0) for r in rows], dtype=np.float32)
    stable = np.array([r.get("stable_score", 0.0) for r in rows], dtype=np.float32)
    texture = np.array([r.get("texture_loss", 0.0) for r in rows], dtype=np.float32)

    score = (
        args.temporal_w_smooth * smooth +
        args.temporal_w_stable * stable +
        args.temporal_w_texture * texture
    )

    return moving_average(np.maximum(score, 0.0), args.temporal_smooth_win)


def max_subarray_segments(values, seed_scores, seed_threshold, min_len=5, max_segments=1):
    values = np.asarray(values, dtype=np.float32)
    seed_scores = np.asarray(seed_scores, dtype=np.float32)

    used = np.zeros(len(values), dtype=bool)
    out = []

    for _ in range(max(1, int(max_segments))):
        best_sum = -1e18
        best_start = None
        best_end = None
        cur_sum = 0.0
        cur_start = 0

        for i, value in enumerate(values):
            if used[i]:
                cur_sum = 0.0
                cur_start = i + 1
                continue

            if cur_sum <= 0.0:
                cur_sum = float(value)
                cur_start = i
            else:
                cur_sum += float(value)

            if cur_sum > best_sum:
                best_sum = cur_sum
                best_start = cur_start
                best_end = i

        if best_start is None or best_end is None:
            break
        if best_sum <= 0.0 or best_end - best_start + 1 < min_len:
            break

        used[best_start:best_end + 1] = True

        # A temporal interval still needs one strong detector seed. This avoids
        # returning a segment on clean videos just because a percentile baseline
        # always leaves some positive cumulative mass.
        if np.max(seed_scores[best_start:best_end + 1]) < seed_threshold:
            continue

        out.append([int(best_start), int(best_end)])

    return sorted(out)


def group_temporal_segments(rows, args):
    if not rows:
        return []

    score = temporal_segment_score(rows, args)
    baseline = np.percentile(score, args.temporal_baseline_percentile)
    values = score - float(baseline)
    seeds = np.array([r.get("smooth_score", 0.0) for r in rows], dtype=np.float32)

    return max_subarray_segments(
        values,
        seeds,
        args.threshold,
        args.min_segment_len,
        args.temporal_max_segments
    )


def select_segments(rows, args):
    threshold_segs = group_segments(
        rows,
        args.threshold,
        args.min_segment_len,
        args.bridge_gap,
        args.low_threshold
    )

    if args.trim_tails:
        threshold_segs = trim_segment_tails(
            rows,
            threshold_segs,
            args.trim_window,
            args.min_segment_len,
            args.trim_score_threshold
        )

    if args.segment_mode == "threshold":
        return threshold_segs

    temporal_segs = group_temporal_segments(rows, args)

    if args.segment_mode == "temporal":
        return temporal_segs

    if not temporal_segs:
        return threshold_segs
    if not threshold_segs:
        return temporal_segs

    temporal_len = segment_length(temporal_segs)
    temporal_frac = temporal_len / float(max(len(rows), 1))

    seed_scores = np.array([r.get("smooth_score", 0.0) for r in rows], dtype=np.float32)
    seed_count = 0

    for start, end in temporal_segs:
        seed_count += int(np.sum(seed_scores[start:end + 1] >= args.threshold))

    seed_density = seed_count / float(max(temporal_len, 1))

    if (
        temporal_frac >= args.temporal_full_video_fraction
        and seed_density < args.temporal_min_seed_density
    ):
        return threshold_segs

    return temporal_segs


def blind_detect_prev_warning(frames, fps, args):
    n = len(frames)

    if n < 5:
        raise ValueError("Need at least 5 frames.")

    gray = [gray01(f) for f in frames]
    bgr = [bgr01(f) for f in frames]

    h, w = gray[0].shape

    patches = make_patches(h, w, args.patch_size, args.stride)

    grid_w = len(range(0, w - args.patch_size + 1, args.stride))
    grid_h = len(range(0, h - args.patch_size + 1, args.stride))

    hp_seq = [highpass(g) for g in gray]
    lap_seq = [lap_abs(g) for g in gray]
    edge_seq = [sobel_mag(g) for g in gray]

    rows = []
    frame_indices = []

    for t in tqdm(range(1, n), desc="Prev-only blind warning-border detection"):
        cur = gray[t]
        prev = gray[t - 1]

        cur_bgr = bgr[t]
        prev_bgr = bgr[t - 1]

        # 1. Align previous frame to current frame
        warp_prev = estimate_affine(prev, cur)

        prev_al = apply_affine(prev, warp_prev, cur.shape)
        prev_bgr_al = apply_affine(prev_bgr, warp_prev, cur_bgr.shape)

        # 2. Current - previous residual maps
        gray_prev_map = np.abs(cur - prev_al)
        rgb_prev_map = np.mean(np.abs(cur_bgr - prev_bgr_al), axis=2)
        hp_prev_map = np.abs(hp_seq[t] - highpass(prev_al))
        edge_prev_map = np.abs(edge_seq[t] - sobel_mag(prev_al))
        lap_prev_map = np.abs(lap_seq[t] - lap_abs(prev_al))

        # 3. Stabilization / texture-loss maps
        s = max(0, t - args.bg_window)
        e = min(n, t + 1)

        recent_stack = np.stack(gray[s:e], axis=0)
        temporal_std = np.std(recent_stack, axis=0)

        stable_map = np.maximum(
            0,
            local_mean(temporal_std, args.local_kernel) - temporal_std
        )

        hp_low_map = local_low(np.abs(hp_seq[t]), args.local_kernel)
        lap_low_map = local_low(lap_seq[t], args.local_kernel)

        # 4. Robust normalized maps
        maps = {
            "rgb_prev": positive_robust_map(rgb_prev_map, 6.0),
            "gray_prev": positive_robust_map(gray_prev_map, 6.0),
            "hp_prev": positive_robust_map(hp_prev_map, 6.0),
            "edge_prev": positive_robust_map(edge_prev_map, 6.0),
            "lap_prev": positive_robust_map(lap_prev_map, 6.0),
            "stable": positive_robust_map(stable_map, 6.0),
            "hp_low": positive_robust_map(hp_low_map, 6.0),
            "lap_low": positive_robust_map(lap_low_map, 6.0),
        }

        # 5. Evidence maps
        transition_map = (
            0.25 * maps["rgb_prev"] +
            0.25 * maps["gray_prev"] +
            0.25 * maps["hp_prev"] +
            0.15 * maps["edge_prev"] +
            0.10 * maps["lap_prev"]
        )

        texture_map = (
            0.45 * maps["stable"] +
            0.35 * maps["hp_low"] +
            0.20 * maps["lap_low"]
        )

        evidence_map = (
            args.map_w_transition * transition_map +
            args.map_w_texture * texture_map
        )

        evidence_map = blur(evidence_map, 5)

        # 6. Patch DCT / patch score
        dct_vals = np.array(
            [dct_hf_ratio(crop(cur, p)) for p in patches],
            dtype=np.float32
        )

        patch_scores = []
        patch_feats = []

        for idx, p in enumerate(patches):
            neigh = neighbor_indices(idx, grid_w, grid_h)
            ref_dct = [dct_vals[j] for j in neigh] if neigh else [dct_vals[idx]]

            dct_low = max(0.0, -robust_z(dct_vals[idx], ref_dct))
            dct_low = min(dct_low, 6.0)

            vals = {k: patch_mean(v, p) for k, v in maps.items()}
            vals["dct_low"] = dct_low

            cue_arr = np.array([
                vals["rgb_prev"],
                vals["gray_prev"],
                vals["hp_prev"],
                vals["edge_prev"],
                vals["stable"],
                vals["hp_low"],
                vals["lap_low"],
                dct_low,
            ], dtype=np.float32)

            strong = np.sum(cue_arr > args.cue_thr)
            medium = np.sum(cue_arr > args.cue_thr * 0.65)

            cooccur = 0.0

            if strong >= 2:
                cooccur += 0.12

            if strong >= 3:
                cooccur += 0.20

            if medium >= 5:
                cooccur += 0.15

            vals["cooccur"] = cooccur

            patch_score = patch_mean(evidence_map, p) + 0.10 * dct_low
            patch_score = patch_score * (1.0 + cooccur)
            patch_score = max(0.0, patch_score)

            patch_scores.append(patch_score)
            patch_feats.append(vals)

        patch_scores = np.asarray(patch_scores, dtype=np.float32)

        # 7. Frame-level features
        row = {
            "frame": int(t),
            "raw_score": safe_float(topk_mean(patch_scores, args.topk)),
        }

        feature_keys = [
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

        for k in feature_keys:
            row[k] = safe_float(topk_mean([pf.get(k, 0.0) for pf in patch_feats], args.topk))

        rows.append(row)
        frame_indices.append(t)

    # =====================================================
    # 8. Series-level scoring
    # =====================================================
    raw = np.array([r["raw_score"] for r in rows], dtype=np.float32)

    # transition: sudden change of frame score
    d_raw = np.abs(np.diff(raw, prepend=raw[0]))
    transition = robust_norm(d_raw, cap=6.0)
    transition = np.maximum(transition, 0)
    transition = moving_average(transition, max(3, args.smooth_win // 2))

    # stable score: smoothed raw score
    raw_norm = robust_norm(raw, cap=6.0)
    raw_norm = np.maximum(raw_norm, 0)
    stable_score = moving_average(raw_norm, args.smooth_win)

    # texture loss score: hp_low + lap_low + dct_low
    texture_arr = np.array([
        r["hp_low"] + r["lap_low"] + r["dct_low"]
        for r in rows
    ], dtype=np.float32)

    texture_loss = robust_norm(texture_arr, cap=6.0)
    texture_loss = np.maximum(texture_loss, 0)
    texture_loss = moving_average(texture_loss, args.smooth_win)

    final = (
        args.w_transition * transition +
        args.w_stable * stable_score +
        args.w_texture * texture_loss
    )

    final = moving_average(final, args.smooth_win)

    for i, r in enumerate(rows):
        r["transition"] = safe_float(transition[i])
        r["stable_score"] = safe_float(stable_score[i])
        r["texture_loss"] = safe_float(texture_loss[i])
        r["score"] = safe_float(final[i])
        r["smooth_score"] = safe_float(final[i])
        r["suspicious"] = bool(final[i] >= args.threshold)

    # =====================================================
    # 9. Segment grouping
    # =====================================================
    seg_ids = select_segments(rows, args)

    segments = []
    valid_mask = np.zeros(len(rows), dtype=bool)

    for s_idx, e_idx in seg_ids:
        valid_mask[s_idx:e_idx + 1] = True

        s_frame = int(rows[s_idx]["frame"])
        e_frame = int(rows[e_idx]["frame"])

        segments.append({
            "start_frame": s_frame,
            "end_frame": e_frame,
            "start_time_sec": safe_float(s_frame / fps),
            "end_time_sec": safe_float(e_frame / fps),
        })

    # 최종 segment로 인정된 프레임만 suspicious=True
    for i, r in enumerate(rows):
        r["suspicious"] = bool(valid_mask[i])

    # =====================================================
    # 10. Final warning border overlay generation
    # =====================================================
    overlay_frames = []

    for i, r in enumerate(rows):
        frame_id = frame_indices[i]
        base_frame = frames[frame_id].copy()

        if r["suspicious"]:
            overlay = make_warning_overlay(
                base_frame,
                text=args.warning_text,
                use_english=args.english_label
            )
        else:
            overlay = base_frame

        overlay_frames.append(overlay)

    result = {
        "mode": "prev_only_blind_warning_border",
        "num_frames": n,
        "fps": fps,
        "frame_scores": rows,
        "suspicious_segments": segments,
    }

    if args.gt_start is not None and args.gt_end is not None:
        result["gt_interval"] = [float(args.gt_start), float(args.gt_end)]

    return result, overlay_frames


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--video", default=None)
    parser.add_argument(
        "--videos",
        nargs="*",
        default=[],
        help="Multiple video paths. Example: --videos a.mp4 b.mp4"
    )
    parser.add_argument(
        "--video_dir",
        action="append",
        default=[],
        help="Directory containing videos. Can be used multiple times."
    )
    parser.add_argument(
        "--video_glob",
        action="append",
        default=[],
        help="Glob pattern for videos. Quote the pattern in the shell."
    )
    parser.add_argument(
        "--dataset_dir",
        action="append",
        default=[],
        help="Dataset root containing case folders with propainter_out.mp4 and original.mp4."
    )
    parser.add_argument("--out_dir", default="./outputs_warning_border")

    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)

    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument(
        "--low_threshold",
        type=float,
        default=None,
        help="Optional hysteresis low threshold. Seeds use --threshold, connected frames can extend down to this score."
    )
    parser.add_argument("--min_segment_len", type=int, default=5)
    parser.add_argument("--bridge_gap", type=int, default=20)
    parser.add_argument(
        "--trim_tails",
        action="store_true",
        help="Trim weak leading/trailing frames from each detected segment."
    )
    parser.add_argument(
        "--trim_window",
        type=int,
        default=9,
        help="Window size for segment tail trimming."
    )
    parser.add_argument(
        "--trim_score_threshold",
        type=float,
        default=0.75,
        help="Local mean smooth_score required to keep segment tails."
    )
    parser.add_argument(
        "--segment_mode",
        choices=["hybrid", "temporal", "threshold"],
        default="hybrid",
        help=(
            "Segment boundary mode. threshold keeps the old threshold/hysteresis "
            "behavior, temporal uses cumulative temporal evidence, and hybrid "
            "uses temporal boundaries with a conservative fallback."
        )
    )
    parser.add_argument(
        "--temporal_baseline_percentile",
        type=float,
        default=40.0,
        help="Percentile baseline for temporal cumulative segment evidence."
    )
    parser.add_argument(
        "--temporal_smooth_win",
        type=int,
        default=13,
        help="Smoothing window for temporal boundary evidence."
    )
    parser.add_argument("--temporal_w_smooth", type=float, default=0.20)
    parser.add_argument("--temporal_w_stable", type=float, default=0.30)
    parser.add_argument("--temporal_w_texture", type=float, default=0.50)
    parser.add_argument(
        "--temporal_max_segments",
        type=int,
        default=1,
        help="Maximum number of temporal suspicious intervals to keep."
    )
    parser.add_argument(
        "--temporal_full_video_fraction",
        type=float,
        default=0.80,
        help="Hybrid fallback starts when temporal coverage is at least this fraction."
    )
    parser.add_argument(
        "--temporal_min_seed_density",
        type=float,
        default=0.25,
        help="Hybrid fallback uses threshold mode when a broad temporal segment has too few threshold seeds."
    )

    parser.add_argument("--smooth_win", type=int, default=13)
    parser.add_argument("--bg_window", type=int, default=5)
    parser.add_argument("--local_kernel", type=int, default=65)

    parser.add_argument("--topk", type=float, default=0.10)
    parser.add_argument("--cue_thr", type=float, default=0.80)

    parser.add_argument("--map_w_transition", type=float, default=0.70)
    parser.add_argument("--map_w_texture", type=float, default=0.30)

    parser.add_argument("--w_transition", type=float, default=0.45)
    parser.add_argument("--w_stable", type=float, default=0.30)
    parser.add_argument("--w_texture", type=float, default=0.25)

    parser.add_argument("--resize_width", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)

    parser.add_argument(
        "--compare_original",
        default=None,
        help="Optional original video for comparison. If omitted, original.mp4 in the same folder is used when present."
    )
    parser.add_argument(
        "--compare_pixel_threshold",
        type=float,
        default=10.0,
        help="A pixel is counted as changed when grayscale abs diff is at least this value."
    )
    parser.add_argument(
        "--compare_ratio_threshold",
        type=float,
        default=0.0001,
        help="A frame is changed when changed pixel ratio is at least this value."
    )
    parser.add_argument(
        "--compare_mean_threshold",
        type=float,
        default=1.0,
        help="A frame is changed when mean grayscale abs diff is at least this value."
    )

    # Warning border label
    parser.add_argument("--warning_text", type=str, default="조작 위험")
    parser.add_argument(
        "--english_label",
        action="store_true",
        help="Use WARNING instead of Korean text. Recommended because OpenCV may not render Korean."
    )

    # Optional graph-only GT display. Not used by detector.
    parser.add_argument("--gt_start", type=float, default=None)
    parser.add_argument("--gt_end", type=float, default=None)

    args = parser.parse_args()

    video_paths = expand_video_inputs(args)

    if not video_paths:
        parser.error("Provide --video, --videos, --video_dir, or --video_glob.")

    ensure_dir(args.out_dir)

    summaries = []

    if len(video_paths) == 1:
        summaries.append(process_video(video_paths[0], args.out_dir, args))
    else:
        used_names = set()

        for idx, video_path in enumerate(video_paths, 1):
            print(f"\n[BATCH {idx}/{len(video_paths)}] {video_path}")
            video_out_dir = unique_output_dir(args.out_dir, video_path, used_names)
            summaries.append(process_video(video_path, video_out_dir, args))

        batch_summary_path = os.path.join(args.out_dir, "batch_summary.json")
        batch_summary_csv = os.path.join(args.out_dir, "batch_summary.csv")
        save_json(batch_summary_path, summaries)
        save_csv(batch_summary_csv, flatten_batch_summary(summaries))
        print("\n[BATCH DONE]")
        print("videos:", len(summaries))
        print("summary:", batch_summary_path)
        print("summary csv:", batch_summary_csv)


if __name__ == "__main__":
    main()
