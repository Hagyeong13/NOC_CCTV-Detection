#!/usr/bin/env python
"""Bake each clip's real manipulated intervals into clips.json (2026-10-13).

The reveal currently says only "this one was ProPainter", which answers the
easy half. The useful comparison is temporal: the model said 0:02-0:05, the
manipulation actually ran 0:03-0:06. That is what makes the demo show
localisation rather than classification.

Ground truth comes from the shared frame label file -- a frame is positive
when its mask has any non-zero pixel -- grouped into maximal runs. The labels
live on the server, so they are copied into clips.json here and travel with
the bundle instead of requiring the 5 MB label CSV.

Note for LASIESTA: its labels are clip-level (every frame of a manipulated
clip is positive) because its masks are unreliable, so its "interval" is the
whole clip. The UI says so rather than implying frame-accurate truth.

    python add_truth_intervals.py
"""
import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
Z_SH = os.path.dirname(HERE)
LABELS = os.path.join(Z_SH, "eval_share", "new_dataset", "frame_labels_ext.csv")
CLIPS = os.path.join(HERE, "clips.json")
MIN_RUN = 1          # ground truth is exact; do not filter it


def runs(frames, labels):
    out, start = [], None
    for i in range(len(labels) + 1):
        on = i < len(labels) and labels[i] == 1
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= MIN_RUN:
                out.append({"start_frame": int(frames[start]),
                            "end_frame": int(frames[i - 1]),
                            "n_frames": int(i - start)})
            start = None
    return out


def main():
    with open(CLIPS, encoding="utf-8") as f:
        payload = json.load(f)
    wanted = {c["id"] for c in payload["clips"]}

    rows = defaultdict(list)
    with open(LABELS, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["case"] in wanted:
                rows[r["case"]].append((int(r["frame"]), int(r["gt_label"])))

    for c in payload["clips"]:
        data = sorted(rows.get(c["id"], []))
        frames = [f for f, _ in data]
        labels = [y for _, y in data]
        segs = runs(frames, labels)
        c["truth_segments"] = segs
        c["truth_frames"] = len(frames)
        c["truth_positive"] = sum(labels)
        # LASIESTA marks whole clips rather than intervals -- flag it so the UI
        # does not present a clip-wide span as frame-accurate localisation
        c["truth_is_cliplevel"] = bool(
            segs and len(segs) == 1 and segs[0]["n_frames"] == len(frames))
        print(f"{c['dataset']:12s} {c['variant']:10s} {len(segs)} interval(s), "
              f"{sum(labels)}/{len(frames)} positive"
              + ("  [clip-level]" if c["truth_is_cliplevel"] else ""))

    with open(CLIPS, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {CLIPS}")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
