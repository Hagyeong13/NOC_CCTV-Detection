#!/usr/bin/env python
"""Graduation-demo UI for the CCTV inpainting-forgery detector (2026-10-12).

Shape of the demo, and why:

  - The evaluator picks a clip from a fixed menu. Every clip comes from the
    held-out test split (see select_demo_clips.py), so "did it memorise
    these?" has an answer. There is no upload box on purpose: the models are
    per-domain and adapted to these five environments, and the package's own
    README measures ~0.51 macro AUC on an unseen domain. Offering arbitrary
    upload would demo the method as something it is not.
  - Each scene appears three times -- pristine, ProPainter, MiniMax -- under
    neutral labels, and the answer stays hidden until Reveal. The model has
    to both flag the manipulated clips and leave the pristine one alone.
  - Output is the clip with the model's own Patch-MIL heatmap burned in, the
    per-frame score curve against the calibrated threshold, the suspicious
    intervals, the verdict, and each branch's calibrated contribution.
  - Results are real inference (~15 s/clip on CPU), cached per clip so the
    showcase is instant. Nothing is fabricated: with no cache and no model
    package reachable, the UI says so rather than inventing a score.

    python app.py                  # 127.0.0.1:7860
    python app.py --share          # public gradio link (72 h)
"""
import argparse
import json
import os

# gradio_client builds a type hint for every API endpoint by walking the
# JSON schema gradio generates. Its walker assumes `additionalProperties` is
# always a sub-schema, but newer pydantic/fastapi emit the boolean form
# (`additionalProperties: true`), and the walker then does `"const" in True`
# -> TypeError: argument of type 'bool' is not iterable, which aborts launch()
# before the server ever binds. Pinning versions only moves the problem
# between releases, so neutralise the walker instead: these hints are used for
# the generated API client, which this demo does not expose.
try:
    import gradio_client.utils as _gcu

    _orig_get_type = _gcu.get_type

    def _get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)

    _gcu.get_type = _get_type

    _orig_json_to_py = _gcu._json_schema_to_python_type

    def _json_to_py(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        try:
            return _orig_json_to_py(schema, defs)
        except TypeError:
            return "Any"

    _gcu._json_schema_to_python_type = _json_to_py
except Exception:                                            # noqa: BLE001
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
Z_SH = os.path.dirname(HERE)
CLIPS_JSON = os.path.join(HERE, "clips.json")
CACHE_DIR = os.path.join(HERE, "cache")
MANIFEST = os.path.join(Z_SH, "eval_share", "new_dataset", "case_manifest_ext.csv")
FPS = 24.0
BRANCH_NAMES = {"lgbm": ("통계 특징", "Statistical"), "compression": ("압축 특징", "Compression"),
                "cnn": ("CNN+시간축", "CNN+TCN"), "patch": ("패치 MIL", "Patch-MIL"),
                "tempfreq": ("시간-주파수", "Temporal-Freq")}
VARIANT_KO = {"pristine": "원본", "propainter": "ProPainter 조작",
              "minimax": "MiniMax 조작"}

# Charts are drawn in one ink plus two greys. Colour is reserved for the
# video overlay, where it carries the only spatial information on screen --
# everywhere else it would compete with it.
INK, MID, FAINT = "#161616", "#5A5A5A", "#D6D6D6"   # --ink / --muted / --idle
LINE, HITSOFT = "#E1E1E1", "#E4E4E4"                 # --line / --hit-soft


def _pick_korean_font():
    """Chart labels are the one place Korean can break: matplotlib draws with
    its own font list, not the browser's. A Windows or Mac demo machine has a
    CJK font; this server has none. Fall back to English labels when it cannot
    be satisfied, rather than drawing boxes."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic",
                 "Noto Sans CJK KR", "Noto Sans KR", "Source Han Sans KR", "UnDotum"):
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


KO_CHARTS = _pick_korean_font()


def t(ko, en):
    return ko if KO_CHARTS else en


with open(CLIPS_JSON, encoding="utf-8") as _f:
    CLIPS = {c["id"]: c for c in json.load(_f)["clips"]}
LABEL_TO_ID = {c["label"]: cid for cid, c in CLIPS.items()}


def mmss(frame):
    s = frame / FPS
    return f"{int(s // 60)}:{s % 60:04.1f}"


def analyze(clip_id):
    """(result dict, overlay mp4 path, error message). Cache hit returns
    immediately; otherwise runs the real pipeline and stores both artefacts."""
    res_path = os.path.join(CACHE_DIR, f"{clip_id}.json")
    vid_path = os.path.join(CACHE_DIR, f"{clip_id}_overlay.mp4")
    if os.path.exists(res_path) and os.path.exists(vid_path):
        with open(res_path, encoding="utf-8") as f:
            return json.load(f), vid_path, None

    try:
        import inference
        import overlay
    except Exception as e:                                   # noqa: BLE001
        return None, None, (f"추론 모듈을 불러오지 못했습니다: {type(e).__name__}: {e}\n\n"
                            "`z-final-sh/standalone_package/`가 있는지 확인하세요.")
    try:
        src = inference.resolve_video(clip_id)
        res = inference.analyze(src, CLIPS[clip_id]["dataset"])
        os.makedirs(CACHE_DIR, exist_ok=True)
        if res.get("heatmap"):
            overlay.render(src, res["heatmap"], vid_path,
                           scores=res["smoothed"], threshold=res["threshold"],
                           segments=res["segments"], show_heat=False)
        res.pop("heatmap", None)          # large; only needed to render
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
    except Exception as e:                                   # noqa: BLE001
        return None, None, f"분석 중 오류: {type(e).__name__}: {e}"
    return res, (vid_path if os.path.exists(vid_path) else None), None


def plot_scores(res):
    frames = np.asarray(res["frames"], dtype=float)
    smoothed = np.asarray(res["smoothed"], dtype=float)
    thr = float(res["threshold"])
    secs = frames / FPS

    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(secs, smoothed, lw=1.7, color=INK,
            label=t("프레임 점수 (평활화)", "frame score (smoothed)"))
    ax.axhline(thr, ls="--", lw=1.1, color=MID,
               label=t(f"임계값 {thr:.2f}", f"threshold {thr:.2f}"))
    for seg in res["segments"]:
        ax.axvspan(seg["start_frame"] / FPS, seg["end_frame"] / FPS,
                   color=HITSOFT, alpha=0.9, zorder=0)
    ax.set_xlabel(t("시간 (초)", "time (s)"))
    ax.set_ylabel(t("위조 점수", "forgery score"))
    ax.set_ylim(0, 1)
    ax.set_xlim(secs.min(), secs.max())
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95,
              edgecolor=LINE)
    ax.grid(alpha=0.9, lw=0.7, color=LINE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def plot_branches(res):
    keys = [k for k in BRANCH_NAMES if k in res.get("branch_means", {})]
    vals = [res["branch_means"][k] for k in keys]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    # one ink, two weights: the branches above the midline read as the ones
    # carrying the call, without introducing a second hue
    colours = [INK if v >= 0.5 else FAINT for v in vals]
    ax.barh([t(*BRANCH_NAMES[k]) for k in keys], vals, color=colours, height=0.62)
    ax.axvline(0.5, ls=":", lw=1.0, color=MID)
    ax.set_xlim(0, 1)
    ax.set_xlabel(t("보정 후 분기 점수", "calibrated branch score"))
    ax.grid(axis="x", alpha=0.9, lw=0.7, color=LINE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def segments_md(res):
    if not res["segments"]:
        return "**의심 구간이 검출되지 않았습니다.**"
    lines = ["| # | 시간 | 프레임 | 평균 | 최고 |", "|---|---|---|---|---|"]
    for i, s in enumerate(res["segments"], 1):
        lines.append(f"| {i} | {mmss(s['start_frame'])}~{mmss(s['end_frame'])} "
                     f"| {s['start_frame']}–{s['end_frame']} ({s['n_frames']}) "
                     f"| {s['mean_score']:.3f} | {s['peak_score']:.3f} |")
    return "\n".join(lines)


def on_analyze(label, progress=gr.Progress()):
    if not label:
        return None, None, None, "클립을 선택해주세요.", "", gr.update(visible=False)
    clip_id = LABEL_TO_ID[label]
    progress(0.1, desc="분석 중 (최초 1회는 약 15초)")
    res, vid, err = analyze(clip_id)
    if err:
        return None, None, None, f"**분석할 수 없습니다.**\n\n{err}", "", gr.update(visible=False)
    progress(0.9, desc="결과 생성")

    flagged = sum(s["n_frames"] for s in res["segments"])
    head = ("### 판정: 조작으로 의심됨" if res["verdict"] == "manipulated"
            else "### 판정: 조작 흔적 없음")
    head += (f"\n\n의심 구간 {len(res['segments'])}개 · "
             f"전체 {len(res['frames'])}프레임 중 {flagged}프레임이 임계값 초과 · "
             f"`{res['domain']}` 도메인 모델({res['checkpoint']}) · "
             f"임계값 {res['threshold']:.3f}"
             + ("(10% FPR 교정)" if res.get("threshold_calibrated") else "(기본값)"))
    return (vid, plot_scores(res), plot_branches(res), head, segments_md(res),
            gr.update(visible=True))


def _spans(segs):
    return ", ".join(f"{mmss(s['start_frame'])}~{mmss(s['end_frame'])}"
                     for s in segs) or "없음"


def on_reveal(label):
    """Reveal is where the demo earns its keep: not just which clip was faked,
    but whether the predicted interval lines up with the real one."""
    if not label:
        return ""
    cid = LABEL_TO_ID[label]
    c = CLIPS[cid]
    truth = "원본 (조작 없음)" if c["truth"] == "clean" else VARIANT_KO[c["variant"]]
    out = [f"### 정답: {truth}",
           f"도메인 `{c['dataset']}` · 장면 `{c['scene']}` · {c['n_frames']}프레임 · "
           f"학습에 쓰이지 않은 **test 분할**", ""]

    res, _, err = analyze(cid)
    pred = res["segments"] if res and not err else []
    true = c.get("truth_segments", [])

    out.append("| | 구간 | 프레임 |")
    out.append("|---|---|---|")
    out.append(f"| **모델 예측** | {_spans(pred)} | "
               f"{sum(s['n_frames'] for s in pred)} / {c['n_frames']} |")
    out.append(f"| **실제 조작** | {_spans(true)} | "
               f"{c.get('truth_positive', 0)} / {c.get('truth_frames', c['n_frames'])} |")

    if c["truth"] == "clean":
        out.append("\n원본이므로 실제 조작 구간이 없습니다. "
                   + ("**모델도 구간을 찾지 않았습니다.**" if not pred
                      else "**모델이 구간을 잘못 표시했습니다(오탐).**"))
    elif c.get("truth_is_cliplevel"):
        out.append("\nLASIESTA는 마스크 신뢰도 문제로 **클립 단위 라벨**을 사용합니다. "
                   "조작 클립의 모든 프레임이 양성으로 표시되어 있어, 구간의 시작과 끝을 "
                   "프레임 단위로 비교할 수는 없습니다.")
    elif pred and true:
        ov = sum(max(0, min(p["end_frame"], t["end_frame"])
                     - max(p["start_frame"], t["start_frame"]) + 1)
                 for p in pred for t in true)
        union = (sum(s["n_frames"] for s in pred)
                 + sum(s["n_frames"] for s in true) - ov)
        out.append(f"\n예측 구간과 실제 구간이 **{ov}프레임 겹칩니다** "
                   f"(IoU {ov / union:.2f}).")
    return "\n".join(out)


CSS = """
/* Ported from the collaborator's design (mono theme): same tokens, same
   radii, same pill buttons. Gradio builds its own DOM, so the structural
   pieces that design does with a CSS grid over the video and a clickable SVG
   chart are not reproducible here -- those live in the static HTML build. */
:root{
  --bg:#EFEFEF; --surface:#FFFFFF; --sunk:#F6F6F6;
  --ink:#161616; --muted:#5A5A5A; --line:#E1E1E1;
  --hit:#1F1F1F; --hit-soft:#E4E4E4; --idle:#D6D6D6;
  --btn:#161616; --btn-ink:#FFFFFF; --chip:#EAEAEA; --chip-ink:#2A2A2A;
  --radius-lg:20px; --radius-md:12px; --radius-sm:8px;
  --font:"Pretendard Variable",Pretendard,-apple-system,"Apple SD Gothic Neo","Malgun Gothic",sans-serif;
}
body, .gradio-container{ background:var(--bg) !important; }
.gradio-container{ max-width:1240px !important; margin:0 auto !important;
  padding:0 clamp(16px,4vw,40px) !important; font-family:var(--font) !important; }
.gradio-container *{ font-family:var(--font) !important; }

h1{ font-size:18px !important; font-weight:700 !important; letter-spacing:-.02em;
    color:var(--ink) !important; margin:22px 0 2px !important; }
h3{ font-size:16px !important; font-weight:700 !important; letter-spacing:-.01em;
    color:var(--ink) !important; }
#intro{ color:var(--muted); font-size:13px; line-height:1.65; margin-bottom:14px; }
#note{ color:var(--muted); font-size:12px; line-height:1.6; margin-top:18px;
       padding-top:14px; border-top:1px solid var(--line); }

/* panels: the design's card, applied to Gradio's block wrappers */
.gr-box, .gr-panel, .gr-form, .block.padded{
  background:var(--surface) !important; border:1px solid var(--line) !important;
  border-radius:var(--radius-lg) !important; box-shadow:none !important; }

/* pill buttons */
button.gr-button, .gr-button{ height:44px !important; padding:0 18px !important;
  border-radius:999px !important; border:1px solid var(--line) !important;
  background:var(--surface) !important; color:var(--ink) !important;
  font-weight:600 !important; box-shadow:none !important; }
button.gr-button:hover{ border-color:var(--muted) !important; }
.gr-button-primary, button.primary{ background:var(--btn) !important;
  color:var(--btn-ink) !important; border-color:var(--btn) !important; }
.gr-button-primary:hover{ opacity:.88 !important; }

select, input, .gr-input, .gr-dropdown{ border-radius:999px !important;
  border:1px solid var(--line) !important; background:var(--surface) !important;
  color:var(--ink) !important; }

#verdict{ background:var(--surface); border:1px solid var(--line);
  border-left:3px solid var(--ink); border-radius:var(--radius-md);
  padding:14px 18px; margin:10px 0 4px; }
#verdict h3{ margin:0 0 6px !important; }

table{ font-size:13px !important; border-collapse:separate !important;
       border-spacing:0 !important; }
table th{ background:var(--sunk) !important; color:var(--muted) !important;
  font-weight:600 !important; border-bottom:1px solid var(--line) !important; }
table td{ border-bottom:1px solid var(--line) !important; }
code{ background:var(--chip) !important; color:var(--chip-ink) !important;
  border-radius:var(--radius-sm) !important; padding:2px 7px !important;
  font-size:12px !important; }
video{ border-radius:var(--radius-md) !important; background:#000 !important; }
.gr-plot, .plot-container{ background:var(--surface) !important;
  border-radius:var(--radius-md) !important; }
footer{ display:none !important; }
"""


def build_ui():
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.gray,
        secondary_hue=gr.themes.colors.gray,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Noto Sans KR"), "Malgun Gothic", "sans-serif"],
    )
    with gr.Blocks(title="CCTV 인페인팅 위조 탐지", theme=theme, css=CSS) as demo:
        gr.Markdown("# CCTV 인페인팅 위조 탐지")
        gr.Markdown(
            "영상에서 객체를 지운 흔적을 프레임 단위로 찾아 의심 구간과 위치를 표시합니다. "
            "같은 장면이 원본 1개와 조작본 2개로 들어 있으며, 어느 쪽인지는 분석 후에 공개됩니다.",
            elem_id="intro")
        with gr.Row():
            clip = gr.Dropdown(sorted(LABEL_TO_ID), label="클립 선택", scale=4)
            btn = gr.Button("분석", variant="primary", scale=1)
        verdict = gr.Markdown(elem_id="verdict")
        with gr.Row():
            video = gr.Video(label="클립 (하단 막대: 프레임별 점수)",
                             scale=3, autoplay=True)
            branches = gr.Plot(label="분기별 기여도", scale=2)
        curve = gr.Plot(label="프레임별 위조 점수")
        table = gr.Markdown()
        reveal_btn = gr.Button("정답 공개", visible=False)
        truth = gr.Markdown()
        gr.Markdown(
            "모든 클립은 학습에 사용되지 않은 test 분할입니다. 모델은 5개 환경에 맞춰 "
            "적응되어 있어 전혀 다른 환경의 영상에는 별도 적응 단계가 필요합니다.",
            elem_id="note")

        btn.click(on_analyze, inputs=clip,
                  outputs=[video, curve, branches, verdict, table, reveal_btn])
        reveal_btn.click(on_reveal, inputs=clip, outputs=truth)
        clip.change(lambda: ("", gr.update(visible=False)), outputs=[truth, reveal_btn])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to reach it from another machine")
    args = ap.parse_args()
    n_cached = len([f for f in os.listdir(CACHE_DIR)
                    if f.endswith(".json")]) if os.path.isdir(CACHE_DIR) else 0
    print(f"{len(CLIPS)} clips, {n_cached} cached, korean charts: {KO_CHARTS}")
    # Bind explicitly, and keep the API schema walk off: show_api=True is what
    # drags gradio_client's type inference in at launch time, which is the path
    # that fails on newer pydantic. The shim at the top of this file covers it
    # too, but not asking for the schema at all is the cheaper fix.
    try:
        build_ui().launch(share=args.share, server_name=args.host,
                          server_port=args.port, show_api=False)
    except OSError as e:
        raise SystemExit(f"port {args.port} is busy ({e}); "
                         f"rerun with --port {args.port + 1}")
