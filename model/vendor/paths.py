"""Filesystem locations for the unified final-release pipeline. Puts the
project's shared feature/branch modules (already carrying the frame-alignment
fix, FRAME_ALIGN_FIXED = True) on sys.path -- this release does NOT vendor a
second copy or a legacy/fixed toggle: there is exactly one, current, fixed
version of each shared module, imported directly.
"""
import os
import sys

ROOT = "/home/tako/songhye/project4"
SHARED_CODE = f"{ROOT}/scripts_2026_07_29"       # forensic_map_encoder_model_a, patch_mil_model,
                                                    # temporal_frequency_model, train_eval_bisfd_pooled_split,
                                                    # compare_weight_models, lodo_fusion -- all fixed 2026-09-22
SPLIT_CODE = f"{ROOT}/scripts_2026_08_20/cafer"    # pooled_split_b2, split_manifest
DCR_CODE = f"{ROOT}/scripts_2026_08_16"            # dcr_net_full_oof (causal_rolling_mean, recall_at_fpr, partial_auc)

for p in (ROOT, SHARED_CODE, SPLIT_CODE, DCR_CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

# main feature CSV -- fixed 2026-09-22: frame-alignment corrected at the
# extraction level, cdnet propainter resolution re-extracted (35 cases),
# shanghaitech/streetscene duplicate-pristine rows removed. 603 case-variants,
# 130,625 rows.
MAIN_CSV = f"{ROOT}/outputs_combined_lasiesta_static/feature_context_report/blob_features/all_domains_codeccontrol_with_compression_rows_EXTENDED.csv"
IITB_CSV = f"{ROOT}/outputs_combined_lasiesta_static/feature_context_report/blob_features/iitb_corridor_full_features.csv"

CODEC_ROOT = f"{ROOT}/codec_controlled_datasets"
CODEC_ROOT_EXTENDED = f"{ROOT}/codec_controlled_datasets_extended"

DOMAINS = ["lasiesta", "shanghaitech", "cdnet", "streetscene", "sbi"]

RESULTS_DIR = f"{ROOT}/final_release/results"


def resolve_video_path(case, dataset):
    """codec-controlled clip for (case, dataset): extended mirror first
    (post-2026-09-04/09-05/09-22 fixes), falling back to the original
    codec-controlled mirror for cases the extension never touched."""
    p_ext = f"{CODEC_ROOT_EXTENDED}/{dataset}/{case}.mp4"
    if os.path.exists(p_ext):
        return p_ext
    return f"{CODEC_ROOT}/{dataset}/{case}.mp4"


def check():
    missing = []
    for name, p in [("MAIN_CSV", MAIN_CSV), ("IITB_CSV", IITB_CSV),
                     ("CODEC_ROOT", CODEC_ROOT), ("CODEC_ROOT_EXTENDED", CODEC_ROOT_EXTENDED)]:
        if not os.path.exists(p):
            missing.append(f"{name}={p}")
    try:
        import forensic_map_encoder_model_a as fmea
        fixed = getattr(fmea, "FRAME_ALIGN_FIXED", False)
    except Exception as e:
        return f"FAILED to import shared branch modules from {SHARED_CODE}: {e}"
    if not fixed:
        missing.append("forensic_map_encoder_model_a.FRAME_ALIGN_FIXED is not True -- "
                        "wrong/stale copy of the shared modules on sys.path")
    if missing:
        return "MISSING:\n  " + "\n  ".join(missing)
    return "OK: all paths resolved, shared modules carry the frame-alignment fix."


if __name__ == "__main__":
    print(check())
