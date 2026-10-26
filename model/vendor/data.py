"""Loads the pooled 5-domain feature CSV and applies the activity-sign
corrections used throughout this project (bisfd.add_signed for
blob_signed/hp_prev_signed; add_signed_compression here for
prnu_min_signed/block_max_signed)."""
import csv

import numpy as np

import paths  # noqa: F401  -- sys.path setup
import compare_weight_models as cw
import train_eval_bisfd_pooled_split as bisfd

DOMAINS = paths.DOMAINS

BASE_FEATS = cw.BASE_FEATURES + ["blob_score", "blob_signed", "hp_prev_signed"]
COMP_FEATS = ["prnu_min", "prnu_mean", "block_max", "block_mean", "prnu_min_signed", "block_max_signed"]
ALL_FEATS = BASE_FEATS + COMP_FEATS


def add_signed_compression(rows):
    by_case = cw.rows_by_case(rows)
    act_by_case = {c: float(np.median(cw.matrix(r, ["rgb_prev"]).ravel())) for c, r in by_case.items()}
    for r in rows:
        sign = 1.0 if act_by_case[r["case"]] > bisfd.ACT_THRESH else -1.0
        r["prnu_min_signed"] = float(cw.safe_float(r.get("prnu_min", 0.0))) * sign
        r["block_max_signed"] = float(cw.safe_float(r.get("block_max", 0.0))) * sign
    return rows


def load_rows(csv_path=None):
    with open(csv_path or paths.MAIN_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["frame"] = int(float(r["frame"]))
        r["_group"] = bisfd.group_key(r["case"])
    rows = bisfd.add_signed(rows)
    rows = add_signed_compression(rows)
    return rows


def load_iitb_rows():
    return load_rows(paths.IITB_CSV)


if __name__ == "__main__":
    rows = load_rows()
    print(f"main: {len(rows)} rows, {len(set(r['case'] for r in rows))} case-variants")
    iitb = load_iitb_rows()
    print(f"iitb: {len(iitb)} rows, {len(set(r['case'] for r in iitb))} case-variants")
