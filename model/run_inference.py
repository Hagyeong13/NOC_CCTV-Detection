#!/usr/bin/env python
"""End-to-end inference: a raw video file -> per-frame manipulation score.
Fully self-contained -- every dependency this needs lives in vendor/ next to
this file, so it runs on any machine with the packages in requirements.txt
installed, with no dependency on the original project4 server tree.

Usage:
  python3 run_inference.py --video /path/to/clip.mp4 --domain sbi
  python3 run_inference.py --video /path/to/clip.mp4 --domain sbi --zero-shot
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PKG_DIR, "vendor"))

import numpy as np
import torch
import lightgbm as lgb
import imageio_ffmpeg

import paths
import data
import fusion
import metrics
import cnn_branch
import patch_branch
import tempfreq_branch
import train_eval_bisfd_pooled_split as bisfd
import compare_weight_models as cw
import forensic_map_encoder_model_a as fmea
import patch_mil_model as pmm
import temporal_frequency_model as tfm
from blob_concentration_feature import blob_scores_for_video
from compression_forensics_features import compression_scores_for_video

# CPU-only inference: no GPU required to run this package.
_CPU = torch.device("cpu")
fmea.DEVICE = _CPU
pmm.DEVICE = _CPU
tfm.DEVICE = _CPU

DETECTOR = os.path.join(_PKG_DIR, "vendor", "blind_inpaint_detector_prev_warning_border.py")
CKPT_ROOT_DEFAULT = os.path.join(_PKG_DIR, "checkpoints")


def normalize_video(src_path, out_path):
    """Same codec normalization as every case in the training dataset: 24fps,
    GOP 24, libx264, CRF 23, yuv420p -- new videos must go through this too so
    feature statistics are on the same footing as the training data."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-y", "-i", src_path, "-r", "24", "-g", "24", "-c:v", "libx264",
           "-crf", "23", "-pix_fmt", "yuv420p", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg normalization failed: {r.stderr[-1500:]}")


def run_detector_csv(video_path, out_dir):
    """Calls the vendored detector script via subprocess (not reimplemented
    in-process), so there is zero risk of drifting from the training-time
    feature definitions."""
    csv_path = os.path.join(out_dir, "frame_scores.csv")
    os.makedirs(out_dir, exist_ok=True)
    r = subprocess.run(
        [sys.executable, DETECTOR, "--video", video_path, "--out_dir", out_dir],
        capture_output=True, text=True,
    )
    if not os.path.exists(csv_path):
        raise RuntimeError(f"detector failed for {video_path}:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return csv_path


def build_rows(video_path, work_dir, case_name="query_video", dataset_tag="query"):
    """Produces this project's row-dict schema (23 features: 17 BASE_FEATS +
    6 COMP_FEATS) for one new video, by calling the exact same extraction
    functions used to build the official training CSV."""
    os.makedirs(work_dir, exist_ok=True)
    mirror_path = os.path.join(work_dir, "mirror.mp4")
    print("[1/4] normalizing video (ffmpeg, 24fps/CRF23/yuv420p)...", flush=True)
    normalize_video(video_path, mirror_path)

    print("[2/4] running base detector (17 stat features)...", flush=True)
    det_out_dir = os.path.join(work_dir, "detector_out")
    csv_path = run_detector_csv(mirror_path, det_out_dir)
    with open(csv_path, encoding="utf-8-sig") as f:
        frame_rows = list(csv.DictReader(f))

    print("[3/4] blob + compression features...", flush=True)
    blob_scores = blob_scores_for_video(mirror_path)
    blob_by_frame = {j + 1: s for j, s in enumerate(blob_scores)}
    comp_scores = compression_scores_for_video(mirror_path)
    n_comp = len(comp_scores["prnu_min"])

    rows = []
    for fr in frame_rows:
        frame_num = int(fr["frame"])
        row = {"case": case_name, "dataset": dataset_tag, "frame": frame_num,
               "gt_label": 0.0, "_group": case_name}
        for k in cw.BASE_FEATURES:
            row[k] = float(fr.get(k, 0.0))
        row["blob_score"] = float(blob_by_frame.get(frame_num, 0.0))
        comp_idx = frame_num - 1
        if 0 <= comp_idx < n_comp:
            for k in ["prnu_min", "prnu_mean", "block_max", "block_mean"]:
                row[k] = float(comp_scores[k][comp_idx])
        else:
            for k in ["prnu_min", "prnu_mean", "block_max", "block_mean"]:
                row[k] = 0.0
        rows.append(row)

    rows = bisfd.add_signed(rows)
    rows = data.add_signed_compression(rows)
    print(f"[4/4] {len(rows)} frame rows built", flush=True)
    return rows, mirror_path


def load_tempfreq_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = tfm.FreqScorer(ckpt["in_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def score_video(video_path, domain, ckpt_root, use_adapted=True, work_dir=None):
    suffix = "adapted" if use_adapted else "zeroshot"
    dom_dir = f"{ckpt_root}/{domain}"
    if not os.path.isdir(dom_dir):
        raise FileNotFoundError(f"no checkpoint dir for domain={domain!r} under {ckpt_root}")

    own_tmp = work_dir is None
    work_dir = work_dir or tempfile.mkdtemp(prefix="inference_")
    rows, mirror_path = build_rows(video_path, work_dir)

    # resolve_video_path override so the branch wrapper modules' own
    # build_all_case_maps(rows) resolve this ONE case to the normalized mirror,
    # regardless of its dummy case/dataset tag.
    def _resolve(case, dataset):
        return mirror_path
    fmea.resolve_video_path = _resolve

    print("scoring: lgbm + compression branches...", flush=True)
    lgbm_booster = lgb.Booster(model_file=f"{dom_dir}/lgbm_{suffix}.txt")
    with open(f"{dom_dir}/lgbm_ctx_keys.json") as f:
        lgbm_ctx_keys = json.load(f)
    X_lgbm, _ = bisfd.build_design(rows, data.BASE_FEATS, lgbm_ctx_keys)
    s_lgbm = lgbm_booster.predict(X_lgbm)

    comp_booster = lgb.Booster(model_file=f"{dom_dir}/compression_{suffix}.txt")
    with open(f"{dom_dir}/compression_ctx_keys.json") as f:
        comp_ctx_keys = json.load(f)
    X_comp, _ = bisfd.build_design(rows, data.COMP_FEATS, comp_ctx_keys)
    s_comp = comp_booster.predict(X_comp)

    print("scoring: cnn (residual) branch...", flush=True)
    cnn_cd = cnn_branch.build_all_case_maps(rows)
    cnn_model = fmea.ModelA()
    cnn_model.load_state_dict(torch.load(f"{dom_dir}/cnn_{suffix}.pt", map_location="cpu"))
    cnn_model.eval()
    s_cnn, _ = cnn_branch.dense_scores_by_row(cnn_model, cnn_cd, rows)

    print("scoring: patch branch...", flush=True)
    patch_cd = patch_branch.build_all_case_maps(rows)
    patch_model = pmm.PatchScorer()
    patch_model.load_state_dict(torch.load(f"{dom_dir}/patch_{suffix}.pt", map_location="cpu"))
    patch_model.eval()
    s_patch, _ = patch_branch.dense_scores_by_row(patch_model, patch_cd, rows)

    print("scoring: tempfreq branch...", flush=True)
    tf_cd = tempfreq_branch.build_all_case_maps(rows)
    tf_model = load_tempfreq_model(f"{dom_dir}/tempfreq_{suffix}.pt")
    if use_adapted:
        with open(f"{dom_dir}/tempfreq_activity_stats.json") as f:
            stats = json.load(f)
        lookup = tempfreq_branch.build_rgb_prev_lookup(rows)
        s_tf, _ = tempfreq_branch.dense_scores_by_row(
            tf_model, tf_cd, rows, activity=(lookup, stats["mean"], stats["std"]))
    else:
        s_tf, _ = tempfreq_branch.dense_scores_by_row(tf_model, tf_cd, rows)

    print("fusing (source-CDF calibration + causal K=32 smoothing)...", flush=True)
    reference = dict(np.load(f"{dom_dir}/fusion_reference.npz"))
    branch_scores = {"lgbm": s_lgbm, "compression": s_comp, "cnn": s_cnn, "patch": s_patch, "tempfreq": s_tf}
    smoothed, calibrated = fusion.fuse(branch_scores, reference, rows)

    frames = [r["frame"] for r in rows]
    if own_tmp:
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)

    return frames, smoothed, branch_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--domain", required=True, choices=data.DOMAINS)
    ap.add_argument("--ckpt-root", default=CKPT_ROOT_DEFAULT)
    ap.add_argument("--zero-shot", action="store_true", help="use the zero-shot (no-DA) checkpoint instead of adapted")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--keep-work-dir", default=None, help="keep intermediate files (mirror video, detector csv) here")
    args = ap.parse_args()

    t0 = time.time()
    frames, smoothed, branch_scores = score_video(
        args.video, args.domain, args.ckpt_root,
        use_adapted=not args.zero_shot, work_dir=args.keep_work_dir)

    n = len(frames)
    flagged = int(np.sum(smoothed >= args.threshold))
    print(f"\n=== result ===")
    print(f"video: {args.video}")
    print(f"domain model: {args.domain} ({'adapted/DA' if not args.zero_shot else 'zero-shot'})")
    print(f"frames: {n}  mean_score={smoothed.mean():.4f}  max_score={smoothed.max():.4f}")
    print(f"frames >= threshold({args.threshold}): {flagged}/{n} ({100*flagged/n:.1f}%)")
    print(f"elapsed: {time.time()-t0:.1f}s")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "score", "flagged"])
            for fr, s in zip(frames, smoothed):
                w.writerow([fr, float(s), int(s >= args.threshold)])
        print(f"per-frame scores written to {args.out_csv}")


if __name__ == "__main__":
    main()
