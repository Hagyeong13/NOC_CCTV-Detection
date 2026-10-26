"""
DCR-Net -- FULL 5-fold cross-fitted OOF base-score verification (2026-08-16).

EXACTLY ONE change from dcr_net_phase1.py (which ADOPTED: macro +0.0099,
worst +0.0006, recall +0.0299, pAUC +0.0087, all 3 seeds consistent):
  l_base for TRAINING rows was computed IN-SAMPLE in Phase 1 (disclosed
  simplification). Here it is replaced with a genuine 5-fold group-based
  cross-fit: the 4 source domains are split into 5 inner folds (by case/
  group), and each fold's rows get a base fusion score from base branches
  trained on the OTHER 4 inner folds only -- every source training row's
  l_base is now 100% out-of-fold.

NOTHING ELSE CHANGES: same RelationEncoder (2 causal dilated Conv1d layers),
same logit-space additive correction (l_final = l_base + gamma*Delta, gamma
learnable, UNBOUNDED -- no tanh clipping), same L_rel-cons domain-consistency
loss, same W=32, same training hyperparameters. Explicitly NOT using the
Relation-Guided Error Correction Net's bounding/ensemble-context/disagreement
additions (that script changed 4 things at once and could not isolate which
one mattered; this script isolates the ONE variable that matters for judging
whether Phase 1's ADOPT verdict was contaminated by in-sample optimism).

Cost: 5 inner-fold trainings + 1 official-pass training = 6x the base-branch
training cost of a normal single-pass run per (domain,seed) -- disclosed
upfront as ~24-27h total for 3-seed x 5-domain, per explicit instruction to
run this exact design regardless.

Same adoption bar as Phase 1: dMacro>=+0.008, worst/recall/pAUC not down,
direction consistent across all 3 seeds.
"""
import csv
import sys

sys.path.insert(0, "/home/tako/songhye/project4/scripts_2026_07_29")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

import lodo_fusion as lf
import forensic_map_encoder_model_a as fmea
import patch_mil_model as pmm
import temporal_frequency_model as tfm
import train_eval_bisfd_pooled_split as bisfd

CSV_PATH = "/home/tako/songhye/project4/outputs_combined_lasiesta_static/feature_context_report/blob_features/all_domains_codeccontrol_with_compression_rows.csv"
CODEC_ROOT = "/home/tako/songhye/project4/codec_controlled_datasets"
DOMAINS = ["lasiesta", "shanghaitech", "cdnet", "streetscene", "sbi"]
SEEDS = [0, 1, 2]
K = 32
W = 32
N_INNER_FOLDS = 5
TARGET_FPR = 0.10
EPS = 1e-4
LAMBDA_REL_CONS = 0.1
REL_LR = 1e-3
REL_EPOCHS = 20
REL_BATCH = 512
COMP_FEATS = ["prnu_min", "prnu_mean", "block_max", "block_mean", "prnu_min_signed", "block_max_signed"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def codec_control_resolve_video_path(case, dataset):
    return f"{CODEC_ROOT}/{dataset}/{case}.mp4"


def load_pool_with_compression():
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["frame"] = int(float(r["frame"]))
    return rows


def add_signed_compression(rows):
    by_case = bisfd.cw.rows_by_case(rows)
    act_by_case = {c: float(np.median(bisfd.cw.matrix(r, ["rgb_prev"]).ravel())) for c, r in by_case.items()}
    for r in rows:
        sign = 1.0 if act_by_case[r["case"]] > bisfd.ACT_THRESH else -1.0
        r["prnu_min_signed"] = float(bisfd.cw.safe_float(r.get("prnu_min", 0.0))) * sign
        r["block_max_signed"] = float(bisfd.cw.safe_float(r.get("block_max", 0.0))) * sign
    return rows


def build_relation_windows(rows):
    by_case = bisfd.cw.rows_by_case(rows)
    eps = 1e-3
    for case, case_rows in by_case.items():
        case_rows.sort(key=lambda r: r["frame"])
        T = np.array([bisfd.cw.safe_float(r.get("transition", 0.0)) for r in case_rows])
        S = np.array([bisfd.cw.safe_float(r.get("stable_score", 0.0)) for r in case_rows])
        medT, medS = np.median(T), np.median(S)
        madT = np.median(np.abs(T - medT)) * 1.4826 + eps
        madS = np.median(np.abs(S - medS)) * 1.4826 + eps
        zT, zS = (T - medT) / madT, (S - medS) / madS
        seq = np.stack([zT, zS], axis=1).astype(np.float32)
        padded = np.pad(seq, ((W - 1, 0), (0, 0)), mode="edge")
        for i, r in enumerate(case_rows):
            r["_relwin"] = padded[i:i + W]
    return rows


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class RelationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = CausalConv1d(2, 16, 3, dilation=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = CausalConv1d(16, 16, 3, dilation=2)
        self.bn2 = nn.BatchNorm1d(16)
        self.head = nn.Linear(16, 1)
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.relu(self.bn1(self.conv1(h)))
        h = F.relu(self.bn2(self.conv2(h)))
        z = h.mean(dim=2)
        delta = self.head(z).squeeze(-1)
        return delta, z


def to_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def train_relation_module(X_rel, l_base_logit, y, domains, seed):
    torch.manual_seed(seed)
    model = RelationEncoder().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=REL_LR)
    bce = nn.BCEWithLogitsLoss()

    Xt = torch.from_numpy(X_rel)
    Lt = torch.from_numpy(l_base_logit)
    Yt = torch.from_numpy(y)
    n = Xt.shape[0]
    domains = np.array(domains)

    model.train()
    for epoch in range(REL_EPOCHS):
        perm = np.random.permutation(n)
        for i in range(0, n, REL_BATCH):
            idx = perm[i:i + REL_BATCH]
            xb = Xt[idx].to(DEVICE)
            lb = Lt[idx].to(DEVICE)
            yb = Yt[idx].to(DEVICE)
            delta, z = model(xb)
            logit = lb + model.gamma * delta
            loss = bce(logit, yb)

            batch_domains = domains[idx]
            batch_y = yb.detach().cpu().numpy()
            group_means = []
            for lab in [0, 1]:
                lab_mask = batch_y == lab
                if lab_mask.sum() < 2:
                    continue
                uniq_d = set(batch_domains[lab_mask])
                if len(uniq_d) < 2:
                    continue
                means_d = []
                for d in uniq_d:
                    m = (batch_domains == d) & (batch_y == lab)
                    if m.sum() == 0:
                        continue
                    means_d.append(z[torch.from_numpy(m).to(DEVICE)].mean(dim=0))
                if len(means_d) >= 2:
                    means_d = torch.stack(means_d)
                    global_mean = means_d.mean(dim=0)
                    group_means.append(((means_d - global_mean) ** 2).sum(dim=1).mean())
            if group_means:
                rel_cons = torch.stack(group_means).mean()
                loss = loss + LAMBDA_REL_CONS * rel_cons

            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def predict_relation_module(model, X_rel, l_base_logit):
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_rel).to(DEVICE)
        Lt = torch.from_numpy(l_base_logit).to(DEVICE)
        delta, _ = model(Xt)
        logit = Lt + model.gamma * delta
        return torch.sigmoid(logit).cpu().numpy()


def percentile_rank(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x))
    return ranks / max(len(x) - 1, 1)


def patch_scores_by_row(model, case_data_subset, rows_in_order):
    X, Y, cases, frames = pmm.make_windows(case_data_subset, stride=1)
    preds = pmm.predict_patch_mil(model, X)
    score_by_key = {(c, f): s for c, f, s in zip(cases, frames, preds)}
    return np.array([score_by_key[(r["case"], int(r["frame"]))] for r in rows_in_order])


def tempfreq_scores_by_row(model, case_data_subset, rows_in_order, sign_by_case):
    X, Y, cases, frames = tfm.make_windows(case_data_subset, stride=1)
    preds = tfm.predict_freq_model(model, X)
    score_by_key = {(c, f): s for c, f, s in zip(cases, frames, preds)}
    scores = np.array([score_by_key[(r["case"], int(r["frame"]))] for r in rows_in_order])
    signs = np.array([sign_by_case[r["case"]] for r in rows_in_order])
    return np.where(signs > 0, scores, 1.0 - scores)


def group_frame_order(rows):
    by_case = {}
    for i, r in enumerate(rows):
        by_case.setdefault(r["case"], []).append(i)
    for c in by_case:
        by_case[c].sort(key=lambda i: int(rows[i]["frame"]))
    return by_case


def causal_rolling_mean(scores, rows, window):
    idx_by_case = group_frame_order(rows)
    out = np.zeros(len(scores))
    for c, idx in idx_by_case.items():
        idx = np.array(idx)
        vals = scores[idx]
        n = len(idx)
        for i in range(n):
            lo = max(0, i - window + 1)
            out[idx[i]] = np.mean(vals[lo:i + 1])
    return out


def recall_at_fpr(y_true, y_score, target_fpr=TARGET_FPR):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def partial_auc(y_true, y_score, max_fpr):
    return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))


def cross_fit_train_base(train_rows, frame_case_data, patch_case_data, freq_case_data, sign_by_case, seed):
    """5-fold group-based cross-fit: EVERY train row gets a base fusion score
    from base branches that never saw its group during training."""
    import random
    groups = sorted(set(bisfd.group_key(r["case"]) for r in train_rows))
    rng = random.Random(seed)
    rng.shuffle(groups)
    fold_of_group = {g: i % N_INNER_FOLDS for i, g in enumerate(groups)}
    row_fold = np.array([fold_of_group[bisfd.group_key(r["case"])] for r in train_rows])

    oof_smooth = np.zeros(len(train_rows), dtype=np.float32)
    for fold_i in range(N_INNER_FOLDS):
        val_mask = row_fold == fold_i
        val_rows = [r for r, m in zip(train_rows, val_mask) if m]
        fit_rows = [r for r, m in zip(train_rows, val_mask) if not m]
        if not val_rows or not fit_rows:
            continue

        fit_cases = set(r["case"] for r in fit_rows)
        val_cases = set(r["case"] for r in val_rows)
        frame_fit = {c: v for c, v in frame_case_data.items() if c in fit_cases}
        frame_val = {c: v for c, v in frame_case_data.items() if c in val_cases}
        patch_fit = {c: v for c, v in patch_case_data.items() if c in fit_cases}
        patch_val = {c: v for c, v in patch_case_data.items() if c in val_cases}
        freq_fit = {c: v for c, v in freq_case_data.items() if c in fit_cases}
        freq_val = {c: v for c, v in freq_case_data.items() if c in val_cases}

        lgbm_v = lf.lgbm_train_predict(fit_rows, val_rows, lf.FEATS, seed=seed)
        comp_v = lf.lgbm_train_predict(fit_rows, val_rows, COMP_FEATS, seed=seed)

        X_f, Y_f, _, _, _ = fmea.make_windows(frame_fit, stride=fmea.STRIDE_TRAIN)
        m_a = fmea.train_model_a(X_f, Y_f)
        cnn_v, _ = lf.cnn_scores_by_row(m_a, frame_val, val_rows)

        X_p, Y_p, _, _ = pmm.make_windows(patch_fit, stride=pmm.STRIDE_TRAIN)
        m_p = pmm.train_patch_mil(X_p, Y_p)
        patch_v = patch_scores_by_row(m_p, patch_val, val_rows)

        X_q, Y_q, _, _ = tfm.make_windows(freq_fit, stride=tfm.STRIDE_TRAIN)
        m_q = tfm.train_freq_model(X_q, Y_q)
        freq_v = tempfreq_scores_by_row(m_q, freq_val, val_rows, sign_by_case)

        fusion_v = (percentile_rank(lgbm_v) + percentile_rank(cnn_v) + percentile_rank(patch_v)
                    + percentile_rank(freq_v) + percentile_rank(comp_v)) / 5.0
        smooth_v = causal_rolling_mean(fusion_v, val_rows, K)
        oof_smooth[np.where(val_mask)[0]] = smooth_v
        print(f"    inner-fold {fold_i+1}/{N_INNER_FOLDS} done "
              f"(fit={len(fit_rows)} val={len(val_rows)})", flush=True)

    return oof_smooth


def run_once(rows, frame_case_data, patch_case_data, freq_case_data, sign_by_case, seed):
    torch.manual_seed(seed)
    results = {}
    for held_out in DOMAINS:
        train_rows = [r for r in rows if r["dataset"] != held_out]
        test_rows = [r for r in rows if r["dataset"] == held_out]

        frame_source = {c: v for c, v in frame_case_data.items() if v[2] != held_out}
        frame_test = {c: v for c, v in frame_case_data.items() if v[2] == held_out}
        patch_source = {c: v for c, v in patch_case_data.items() if v[2] != held_out}
        patch_test = {c: v for c, v in patch_case_data.items() if v[2] == held_out}
        freq_source = {c: v for c, v in freq_case_data.items() if v[2] != held_out}
        freq_test = {c: v for c, v in freq_case_data.items() if v[2] == held_out}

        # official pass: fit on ALL 4 source domains, score the held-out test domain
        lgbm_scores = lf.lgbm_train_predict(train_rows, test_rows, lf.FEATS, seed=seed)
        comp_scores = lf.lgbm_train_predict(train_rows, test_rows, COMP_FEATS, seed=seed)

        X_f, Y_f, _, _, _ = fmea.make_windows(frame_source, stride=fmea.STRIDE_TRAIN)
        model_a = fmea.train_model_a(X_f, Y_f)
        cnn_scores, cnn_y = lf.cnn_scores_by_row(model_a, frame_test, test_rows)

        X_p, Y_p, _, _ = pmm.make_windows(patch_source, stride=pmm.STRIDE_TRAIN)
        model_p = pmm.train_patch_mil(X_p, Y_p)
        patch_scores = patch_scores_by_row(model_p, patch_test, test_rows)

        X_q, Y_q, _, _ = tfm.make_windows(freq_source, stride=tfm.STRIDE_TRAIN)
        model_q = tfm.train_freq_model(X_q, Y_q)
        freq_scores = tempfreq_scores_by_row(model_q, freq_test, test_rows, sign_by_case)

        y_test = bisfd.cw.labels(test_rows).astype(int)
        assert np.array_equal(y_test, cnn_y.astype(int))

        fusion_test = (percentile_rank(lgbm_scores) + percentile_rank(cnn_scores) + percentile_rank(patch_scores)
                       + percentile_rank(freq_scores) + percentile_rank(comp_scores)) / 5.0
        smooth_test_base = causal_rolling_mean(fusion_test, test_rows, K)

        # THE ONE CHANGE: 5-fold cross-fitted OOF base score for training rows
        print(f"  [seed={seed}] {held_out}: cross-fitting train base scores...", flush=True)
        smooth_train_base = cross_fit_train_base(
            train_rows, frame_case_data, patch_case_data, freq_case_data, sign_by_case, seed)
        y_train = bisfd.cw.labels(train_rows).astype(np.float32)

        l_base_train = to_logit(smooth_train_base).astype(np.float32)
        l_base_test = to_logit(smooth_test_base).astype(np.float32)
        X_rel_train = np.stack([r["_relwin"] for r in train_rows]).astype(np.float32)
        X_rel_test = np.stack([r["_relwin"] for r in test_rows]).astype(np.float32)
        domain_train = [r["dataset"] for r in train_rows]

        rel_model = train_relation_module(X_rel_train, l_base_train, y_train, domain_train, seed)
        final_test = predict_relation_module(rel_model, X_rel_test, l_base_test)

        metrics = {}
        for name, s in [("base", smooth_test_base), ("dcr", final_test)]:
            metrics[name] = {
                "auc": roc_auc_score(y_test, s),
                "recall": recall_at_fpr(y_test, s),
                "pauc": partial_auc(y_test, s, 0.1),
            }
        results[held_out] = metrics
        print(f"  [seed={seed}] {held_out}: base(AUC={metrics['base']['auc']:.4f} "
              f"R@.1={metrics['base']['recall']:.4f} pAUC={metrics['base']['pauc']:.4f}) | "
              f"DCR(AUC={metrics['dcr']['auc']:.4f} R@.1={metrics['dcr']['recall']:.4f} "
              f"pAUC={metrics['dcr']['pauc']:.4f}) gamma={rel_model.gamma.item():.4f}", flush=True)
    return results


def main():
    fmea.resolve_video_path = codec_control_resolve_video_path
    print("resolve_video_path monkeypatched -> codec_controlled_datasets/", flush=True)
    print(f"W={W} K={K} N_INNER_FOLDS={N_INNER_FOLDS} (100% OOF cross-fit, no bounding, no context -- "
          f"the ONLY change vs dcr_net_phase1.py)", flush=True)

    rows = load_pool_with_compression()
    rows = bisfd.add_signed(rows)
    rows = add_signed_compression(rows)
    sign_by_case = tfm.case_activity_signs(rows)

    print("building causal relation windows (transition, stable_score)...", flush=True)
    rows = build_relation_windows(rows)

    print("extracting frame-level maps...", flush=True)
    frame_case_data = fmea.build_case_maps(rows)
    print("extracting patch-level maps...", flush=True)
    patch_case_data = pmm.build_case_patch_maps(rows)
    print("extracting patch-signal series...", flush=True)
    freq_case_data = tfm.build_case_patch_signals(rows)

    per_seed = []
    for seed in SEEDS:
        print(f"\n########## seed={seed} ##########", flush=True)
        per_seed.append(run_once(rows, frame_case_data, patch_case_data, freq_case_data, sign_by_case, seed))

    print("\n" + "=" * 90)
    print("=== 3-seed domain summary ===", flush=True)
    domain_avg = {d: {"base": {}, "dcr": {}} for d in DOMAINS}
    for d in DOMAINS:
        for metric in ["auc", "recall", "pauc"]:
            for variant in ["base", "dcr"]:
                vals = [res[d][variant][metric] for res in per_seed]
                domain_avg[d][variant][metric] = np.mean(vals)
        b, t = domain_avg[d]["base"], domain_avg[d]["dcr"]
        print(f"  {d}: AUC base={b['auc']:.4f} dcr={t['auc']:.4f} (d={t['auc']-b['auc']:+.4f}) | "
              f"R@.1 base={b['recall']:.4f} dcr={t['recall']:.4f} (d={t['recall']-b['recall']:+.4f}) | "
              f"pAUC base={b['pauc']:.4f} dcr={t['pauc']:.4f} (d={t['pauc']-b['pauc']:+.4f})", flush=True)

    seed_macro_base_auc = [np.mean([res[d]["base"]["auc"] for d in DOMAINS]) for res in per_seed]
    seed_macro_dcr_auc = [np.mean([res[d]["dcr"]["auc"] for d in DOMAINS]) for res in per_seed]
    seed_worst_base_auc = [min(res[d]["base"]["auc"] for d in DOMAINS) for res in per_seed]
    seed_worst_dcr_auc = [min(res[d]["dcr"]["auc"] for d in DOMAINS) for res in per_seed]
    seed_macro_base_recall = [np.mean([res[d]["base"]["recall"] for d in DOMAINS]) for res in per_seed]
    seed_macro_dcr_recall = [np.mean([res[d]["dcr"]["recall"] for d in DOMAINS]) for res in per_seed]
    seed_macro_base_pauc = [np.mean([res[d]["base"]["pauc"] for d in DOMAINS]) for res in per_seed]
    seed_macro_dcr_pauc = [np.mean([res[d]["dcr"]["pauc"] for d in DOMAINS]) for res in per_seed]

    macro_base_auc, macro_dcr_auc = np.mean(seed_macro_base_auc), np.mean(seed_macro_dcr_auc)
    worst_base_auc, worst_dcr_auc = np.mean(seed_worst_base_auc), np.mean(seed_worst_dcr_auc)
    macro_base_recall, macro_dcr_recall = np.mean(seed_macro_base_recall), np.mean(seed_macro_dcr_recall)
    macro_base_pauc, macro_dcr_pauc = np.mean(seed_macro_base_pauc), np.mean(seed_macro_dcr_pauc)

    n_domains_up = sum(1 for d in DOMAINS if domain_avg[d]["dcr"]["auc"] > domain_avg[d]["base"]["auc"])
    direction_consistent = all(seed_macro_dcr_auc[i] > seed_macro_base_auc[i] for i in range(len(SEEDS)))

    print(f"\nmacro AUC:    base={macro_base_auc:.4f} dcr={macro_dcr_auc:.4f} delta={macro_dcr_auc-macro_base_auc:+.4f}")
    print(f"worst AUC:    base={worst_base_auc:.4f} dcr={worst_dcr_auc:.4f} delta={worst_dcr_auc-worst_base_auc:+.4f}")
    print(f"macro Recall@.1FPR: base={macro_base_recall:.4f} dcr={macro_dcr_recall:.4f} delta={macro_dcr_recall-macro_base_recall:+.4f}")
    print(f"macro pAUC(.1): base={macro_base_pauc:.4f} dcr={macro_dcr_pauc:.4f} delta={macro_dcr_pauc-macro_base_pauc:+.4f}")
    print(f"domains AUC improved: {n_domains_up}/5")
    print(f"direction consistent across all {len(SEEDS)} seeds: {direction_consistent}")

    c1 = (macro_dcr_auc - macro_base_auc) >= 0.008
    c2 = worst_dcr_auc >= worst_base_auc
    c3 = macro_dcr_recall >= macro_base_recall
    c4 = macro_dcr_pauc >= macro_base_pauc
    c5 = direction_consistent
    verdict = "ADOPT" if (c1 and c2 and c3 and c4 and c5) else "REJECT"
    print(f"\ncriteria: dMacro>=0.008={c1} worst_not_down={c2} recall_not_down={c3} pAUC_not_down={c4} "
          f"direction_consistent={c5}")
    print(f"FINAL VERDICT: {verdict}")
    if verdict == "REJECT" and c1 is False:
        print("If Phase 1's dMacro (+0.0099) shrinks substantially here, the in-sample l_base_train "
              "was likely inflating Phase 1's result -- DCR-Net's main claim should be withdrawn.")
    print("\nALL_DONE", flush=True)


if __name__ == "__main__":
    main()