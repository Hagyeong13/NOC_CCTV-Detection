#!/usr/bin/env python
"""Burn the model's Patch-MIL heatmap onto the clip (2026-10-12).

What is drawn is the 4x4 per-patch score the model itself produced -- see
inference.py -- upsampled smoothly and tinted. It localises to about a
sixteenth of the frame; it is not a segmentation mask, and the caption drawn
on the video says as much so a still frame cannot be mistaken for one.

Frame alignment matters here and is easy to get wrong. Patch extraction drops
the clip's first frame (`frames_patches[1:]`), and this project's convention
is that CSV row f corresponds to video frame f+1. So heatmap[i] belongs on
the video frame at 0-based position i+1, and video frame 0 gets no overlay.

Alpha follows the score, so cold regions keep their original pixels instead
of the whole frame taking on a tint.
"""
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

ALPHA_MAX = 0.55          # opacity at the clip's strongest deviation
BAR_H = 34
TOP_K = 2                 # patches painted per frame, of 16


def _prepare(heatmap, scores, threshold):
    """Turn raw patch scores into something that means what the picture implies.

    Three corrections, each for a different way the raw map misleads:

    1. Patch-MIL scores rank patches against each other, so the patch holding a
       moving person is highest even in untouched video. Subtracting each
       patch's own baseline -- its median over the frames the model did NOT
       flag -- removes "this corner is always busy" and leaves "this is odd for
       this scene". Falls back to the global median when nothing is unflagged.
    2. The model's decision uses only its top instances, so painting every
       patch over a fixed floor overstates what drove the score. Keep TOP_K.
    3. Scale by the clip's own strongest deviation, so the display does not
       depend on absolute logit values that vary by domain."""
    heat = np.asarray(heatmap, dtype=np.float32)
    n = len(heat)
    if scores is not None and len(scores) >= n:
        quiet = np.asarray(scores[:n], dtype=float) < threshold
    else:
        quiet = np.ones(n, dtype=bool)
    ref = np.median(heat[quiet], axis=0) if quiet.any() else np.median(heat, axis=0)

    adj = np.clip(heat - ref[None], 0.0, None)
    flat = adj.reshape(n, -1)
    if flat.shape[1] > TOP_K:
        cut = np.partition(flat, -TOP_K, axis=1)[:, -TOP_K][:, None]
        flat = np.where(flat >= cut, flat, 0.0)
    adj = flat.reshape(heat.shape)
    peak = float(adj.max())
    return adj / peak if peak > 1e-6 else adj


# One warm ramp, yellow to red, with opacity carrying the magnitude. A
# rainbow map (JET) paints low-confidence regions a vivid blue, which reads as
# a finding rather than as "nothing here" -- the wrong impression for a demo.
_LOW = np.array([0, 210, 255], dtype=np.float32)    # BGR yellow
_HIGH = np.array([0, 0, 235], dtype=np.float32)     # BGR red


def _tint(heat_up):
    """Yellow -> red by heat value, as BGR uint8 of shape (h, w, 3)."""
    k = np.clip(heat_up, 0.0, 1.0)[:, :, None]
    return (_LOW * (1 - k) + _HIGH * k).astype(np.uint8)


def render(video_path, heatmap, out_path, scores=None, threshold=0.5, fps=None,
           segments=None, show_heat=False):
    """Write an mp4 with the heatmap blended in. `heatmap` is (n, 4, 4) in
    [0,1]; frames past its end are written through unchanged."""
    # The 4x4 map localises to a sixteenth of the frame, which reads as a
    # claim the model cannot support; the score curve and the interval table
    # carry "when" without implying a "where" this precise. Off by default.
    heatmap = _prepare(heatmap, scores, threshold) if show_heat else None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps or (cap.get(cv2.CAP_PROP_FPS) or 24.0)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h + BAR_H))
    scores = None if scores is None else np.asarray(scores, dtype=float)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hi = i - 1                       # video frame 0 has no heatmap
        # Only paint frames the model actually flagged. The Patch-MIL logits
        # are a RELATIVE ranking of patches -- on a clean clip the patch holding
        # a moving person still scores highest, so drawing them unconditionally
        # made the heatmap trail people through pristine video while the verdict
        # said "clean". Gating on the frame score keeps the picture and the
        # verdict telling the same story.
        # Inside an accepted interval, not merely over the threshold. A clean
        # clip can cross for a few frames without the run being long enough to
        # report; painting those would contradict the verdict on screen.
        if segments:
            flagged = any(s["start_frame"] <= hi + 1 <= s["end_frame"]
                          for s in segments)
        else:
            flagged = (scores is not None and 0 <= hi < len(scores)
                       and float(scores[hi]) >= threshold)
        if show_heat and heatmap is not None and 0 <= hi < len(heatmap) and flagged:
            heat_up = cv2.resize(heatmap[hi], (w, h), interpolation=cv2.INTER_CUBIC)
            heat_up = np.clip(heat_up, 0.0, 1.0)
            colour = _tint(heat_up).astype(np.float32)
            a = (heat_up * ALPHA_MAX)[:, :, None]
            frame = (frame.astype(np.float32) * (1 - a) + colour * a).astype(np.uint8)

        canvas = np.zeros((h + BAR_H, w, 3), dtype=np.uint8)
        canvas[:h] = frame
        if scores is not None and 0 <= hi < len(scores):
            s = float(scores[hi])
            flagged = s >= threshold
            cv2.rectangle(canvas, (0, h), (int(w * min(s, 1.0)), h + BAR_H - 14),
                          (60, 60, 220) if flagged else (150, 150, 150), -1)
            cv2.putText(canvas, f"score {s:.2f}  frame {hi + 1}", (6, h + BAR_H - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if show_heat:
            cv2.putText(canvas, "Patch-MIL 4x4 heatmap (coarse region, not a mask)",
                        (6, h + BAR_H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(canvas)
        i += 1

    cap.release()
    writer.release()
    return _to_h264(out_path)


def _to_h264(path):
    """OpenCV can only write mp4v here (its pip build ships no H.264 encoder),
    and browsers will not play mp4v -- the video element silently shows
    nothing, which is exactly how this surfaced. imageio-ffmpeg is already a
    dependency and bundles an ffmpeg with libx264, so transcode in place.
    If that is unavailable the mp4v file is left as-is rather than lost."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                        # noqa: BLE001
        return path
    tmp = path + ".h264.mp4"
    cmd = [exe, "-y", "-loglevel", "error", "-i", path,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           "-movflags", "+faststart", tmp]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception:                                        # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        return path
    shutil.move(tmp, path)
    return path
