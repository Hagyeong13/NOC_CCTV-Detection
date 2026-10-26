#!/usr/bin/env python
"""Render the demo into the collaborator's HTML design (2026-10-15).

The design that arrived is not a mock-up, it is a working front end, and its
structure beats what the Gradio version could do: the heatmap is a DOM grid
over the <video> rather than pixels burned into it, so it toggles on and off
and needs no re-encoding; the score curve is an SVG you can click to seek; and
the reveal already draws predicted and true intervals as comparable bars.

It also declares a clean data contract -- one DATA object per clip -- so this
script only has to fill it from the cache and swap the upload control for a
picker over our fixed clip menu. The result is one self-contained page: no
Python, no pip, no port at the showcase.

Two adaptations to the template:
  - upload / analyze are replaced by a clip <select>, because the models are
    per-domain and arbitrary upload would misrepresent the method
  - the heatmap is the per-clip aggregate (see inference.analyze): one 4x4 map
    for the clip, whose intensity the template already modulates by the
    current frame's score

    python build_html_demo.py                 # -> html_demo/index.html
    python build_html_demo.py --open-note
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
Z_SH = os.path.dirname(HERE)
TEMPLATE = os.path.join(Z_SH, "incoming", "cctv_tamper_detect.html")
CACHE = os.path.join(HERE, "cache")
OUT_DIR = os.path.join(HERE, "html_demo")
BRANCH_KO = {"lgbm": "통계 특징", "compression": "압축 특징", "cnn": "CNN+시간축",
             "patch": "패치 MIL", "tempfreq": "시간-주파수"}
VARIANT_KO = {"pristine": "원본 (조작 없음)", "propainter": "ProPainter 조작",
              "minimax": "MiniMax 조작"}
FPS = 24


def _norm(heat):
    if not heat:
        return [0.0] * 16
    peak = max(heat)
    return [round(v / peak, 4) for v in heat] if peak > 1e-6 else [0.0] * 16


def clip_data(clip, res):
    segs = [{"start": s["start_frame"], "end": s["end_frame"],
             "mean": round(s["mean_score"], 4), "max": round(s["peak_score"], 4)}
            for s in res["segments"]]
    truth_segs = clip.get("truth_segments", [])
    n = len(res["frames"])
    peak = max(range(n), key=lambda i: res["smoothed"][i]) + 1
    return {
        "id": clip["id"],
        "file": clip["id"] + ".mp4",
        "label": clip["label"],
        "frames": n,
        "fps": FPS,
        "threshold": round(float(res["threshold"]), 4),
        "scores": [round(float(x), 4) for x in res["smoothed"]],
        "segments": segs,
        "branches": [{"name": BRANCH_KO.get(k, k), "score": round(v, 4)}
                     for k, v in res.get("branch_means", {}).items()],
        # normalised so the clip's strongest patch is 1.0: the template's
        # "hot" test is an absolute 0.3, and averaging top-2 masks over the
        # flagged frames leaves raw values well under that, so without this
        # most clips would draw no overlay at all
        "heatmap": _norm(res.get("heatmap_agg")),
        "truth": {
            "label": VARIANT_KO.get(clip["variant"], clip["variant"]),
            "domain": clip["dataset"], "scene": clip["scene"], "split": "test",
            "start": truth_segs[0]["start_frame"] if truth_segs else 0,
            "end": truth_segs[-1]["end_frame"] if truth_segs else 0,
            "clean": clip["truth"] == "clean",
            "cliplevel": bool(clip.get("truth_is_cliplevel")),
        },
        "startFrame": peak,
    }


PICKER = '''      <label class="btn" for="clipPick" style="cursor:pointer">클립</label>
      <select id="clipPick" class="btn" style="min-width:260px;cursor:pointer"></select>
'''

LOADER = '''
/* ---- clip picker: the menu is fixed because each model is adapted to one
   domain; arbitrary upload would demo something the method does not claim ---- */
function loadClip(id) {
  DATA = CLIPS[id];
  N = DATA.frames; FPS = DATA.fps; frame = DATA.startFrame;
  const v = $("video");
  v.pause(); v.removeAttribute("poster"); v.src = "videos/" + DATA.file;
  $("fileMeta").textContent = DATA.label + " · " + DATA.frames + " 프레임 · " + DATA.fps + "fps";
  $("answer").classList.remove("open");
  $("reveal").setAttribute("aria-expanded", false);
  $("revealLabel").textContent = "정답 공개";
  drawChart(); drawBranches(); drawSummary(); drawAnswer(); setFrame(frame);
}
const sel = $("clipPick");
sel.innerHTML = Object.keys(CLIPS).map(k =>
  `<option value="${k}">${CLIPS[k].label}</option>`).join("");
sel.addEventListener("change", e => loadClip(e.target.value));
loadClip(Object.keys(CLIPS)[0]);
'''


def _find_ffmpeg():
    """imageio-ffmpeg is not installed in every env this script gets run from,
    but its bundled binary usually exists in one of the project's conda envs.
    Fall back to PATH, then to those bundles, before giving up."""
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    pats = ("/data/cache/conda/*/envs/*/lib/python*/site-packages/"
            "imageio_ffmpeg/binaries/ffmpeg-linux*",
            os.path.expanduser("~/.conda/envs/*/lib/python*/site-packages/"
                               "imageio_ffmpeg/binaries/ffmpeg-linux*"))
    cands = [c for pat in pats for c in glob.glob(pat) if os.access(c, os.X_OK)]
    # newest build last by name, which sorts v4 < v7
    return sorted(cands)[-1] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    import inference

    with open(os.path.join(HERE, "clips.json"), encoding="utf-8") as f:
        clips = json.load(f)["clips"]

    data = {}
    for c in clips:
        p = os.path.join(CACHE, f"{c['id']}.json")
        if not os.path.exists(p):
            print(f"  {c['id']}: no cache -- skipped")
            continue
        with open(p, encoding="utf-8") as f:
            res = json.load(f)
        if not res.get("heatmap_agg"):
            print(f"  {c['id']}: no heatmap_agg -- rerun build_cache.py --force")
        data[c["id"]] = clip_data(c, res)
    if not data:
        sys.exit("no cached clips")

    with open(TEMPLATE, encoding="utf-8") as f:
        html = f.read()

    # single-clip DATA -> all clips, and the demo-curve fallback goes away
    html = re.sub(r"const DATA = \{.*?\n\};\n", "", html, flags=re.S)
    html = re.sub(r"if \(!DATA\.scores\) \{.*?\n\}\n", "", html, flags=re.S)
    html = html.replace(
        "const $ = (id) => document.getElementById(id);",
        "const CLIPS = " + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        + ";\nlet DATA = CLIPS[Object.keys(CLIPS)[0]];\n"
        "const $ = (id) => document.getElementById(id);")
    html = html.replace("const N = DATA.frames, FPS = DATA.fps;",
                        "let N = DATA.frames, FPS = DATA.fps;")

    # upload + analyze -> picker
    html = re.sub(r'      <label class="btn" for="upload">.*?id="analyze">분석 실행</button>\n',
                  PICKER, html, flags=re.S)
    html = re.sub(r'\$\("upload"\)\.addEventListener.*?\n\}\);\n', "", html, flags=re.S)
    html = re.sub(r'\$\("analyze"\)\.addEventListener.*?\n\}\);\n', "", html, flags=re.S)

    # pristine clips have no predicted interval; the template assumes one
    html = html.replace("const T = DATA.truth, P = DATA.segments[0];",
                        'const T = DATA.truth;\n'
                        '  const P = DATA.segments[0] || {start: 0, end: -1};')
    html = html.replace('$("ansNote").innerHTML = `예측 구간과',
                        'if (T.clean) { $("ansNote").innerHTML = P.end >= P.start\n'
                        '    ? "원본인데 모델이 구간을 표시했습니다 (오탐)."\n'
                        '    : "원본이고, 모델도 구간을 찾지 않았습니다."; return; }\n'
                        '  if (T.cliplevel) { $("ansNote").innerHTML = '
                        '"LASIESTA는 마스크 신뢰도 문제로 <strong>클립 단위 라벨</strong>을 사용합니다. '
                        '조작 클립의 모든 프레임이 양성이라 구간의 시작과 끝을 프레임 단위로 비교할 수 없습니다."; return; }\n'
                        '  $("ansNote").innerHTML = `예측 구간과')

    # fixed mono: drop the theme switcher markup and its handler
    html = html.replace('data-theme="pastel"', 'data-theme="mono"', 1)
    html = re.sub(r'      <div class="theme" role="group".*?</div>\n', "", html, flags=re.S)
    html = re.sub(r'/\* theme \*/\ndocument\.querySelectorAll\(\"\[data-theme-set\]\"\).*?\}\)\);\n',
                  "", html, flags=re.S)

    # Heat overlay removed: a 4x4 grid cannot honestly indicate "where", and
    # on screen it invited exactly that reading. "When" is carried by the score
    # curve and the interval table, which the model does support.
    html = re.sub(r'<div class="grid" id="heat".*?</div>\n', "", html, flags=re.S)
    html = re.sub(r'<button type="button" class="switch" id="heatToggle".*?</button>\n',
                  "", html, flags=re.S)
    html = re.sub(r'function drawHeat\(\) \{.*?\n\}\n', "", html, flags=re.S)
    html = html.replace("drawHeat();", "")
    # Line-anchored, NOT DOTALL: this listener is a one-liner, so a `.*?\n});\n`
    # pattern runs past it to the next multi-line block's closing brace and
    # silently takes drawBranches/drawSummary/drawAnswer with it.
    html = re.sub(r'^\$\("heatToggle"\)\.addEventListener.*\n', "", html, flags=re.M)


    # drawSummary fills the stat row only when a clip is flagged, so switching
    # from a forged clip to a clean one left the previous clip's numbers on
    # screen beside a "정상으로 보여요" verdict. Blank them instead.
    html = html.replace(
        """  if (flagged) {
    const top = segs.reduce((a, b) => b.max > a.max ? b : a);""",
        """  if (!flagged) {
    $("sMax").textContent = Math.max(...DATA.scores).toFixed(3);
    $("sMean").textContent = "\u2013";
    $("sRange").textContent = "\u2013";
  }
  if (flagged) {
    const top = segs.reduce((a, b) => b.max > a.max ? b : a);""")

    # the heat overlay is gone; its toggle state variable is now unused
    html = html.replace("let frame = DATA.startFrame, heatOn = true, timer = null;",
                        "let frame = DATA.startFrame, timer = null;")

    # The panel header still advertised the overlay that was just removed, so
    # the page promised a "where" it no longer shows. The panel is now the clip
    # plus the timeline, and the heading says that instead.
    # a comment introducing the overlay routine that is no longer emitted
    html = html.replace(
        "/* heatmap: 현재 프레임 점수에 비례해 강도 조절 */\n\n", "")
    html = html.replace('<h2 id="locTitle">탐지 위치</h2>',
                        '<h2 id="locTitle">영상 확인</h2>')
    html = html.replace(
        '<span class="sub">Patch-MIL 4×4 히트맵, 대략적인 영역만 표시</span>',
        '<span class="sub">조작 의심 구간이 타임라인에 표시됩니다</span>')

    html = html.replace("drawChart(); drawBranches(); drawSummary(); drawAnswer(); setFrame(frame);",
                        LOADER)

    # The edits above are regex surgery on someone else's file; a pattern that
    # over-matches removes working code without erroring. Check the functions
    # the page needs are all still here before writing anything out.
    required = ("drawChart", "drawBranches", "drawSummary", "drawAnswer",
                "setFrame", "loadClip")
    missing = [f for f in required if f"function {f}" not in html]
    if missing:
        sys.exit(f"template edits removed: {', '.join(missing)} -- aborting")
    for gone in ("drawHeat", "heatToggle"):
        if gone in html:
            sys.exit(f"{gone} should have been removed but is still present")

    out = os.path.abspath(args.out)
    # Rebuild the video folder from scratch: clips dropped from the menu would
    # otherwise linger from an earlier build and ship in the bundle.
    vid_dir = os.path.join(out, "videos")
    if os.path.isdir(vid_dir):
        shutil.rmtree(vid_dir)
    os.makedirs(vid_dir, exist_ok=True)
    # The dataset clips are written with the extension pipeline's mp4v writer,
    # and a browser's <video> will not decode mpeg4 -- it fetches the file
    # (200 in the server log) and then shows nothing, which is how this
    # surfaced. Transcode to H.264 on the way into the bundle; copy only what
    # is already H.264.
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                        # noqa: BLE001
        ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        # Copying the sources instead is worse than failing: six of the ten
        # dataset clips are mp4v, which a browser refuses to decode while the
        # page itself loads fine -- so the demo looks built and plays nothing.
        sys.exit("no ffmpeg found; cannot transcode to H.264. "
                 "Install imageio-ffmpeg or put ffmpeg on PATH.")
    n_conv = 0
    for cid in data:
        src = inference.resolve_video(cid)
        dst = os.path.join(out, "videos", f"{cid}.mp4")
        codec = ""
        if ffmpeg:
            info = subprocess.run([ffmpeg, "-i", src], capture_output=True,
                                  text=True).stderr
            line = [l for l in info.splitlines() if "Video:" in l]
            codec = line[0].split("Video: ")[1].split()[0] if line else ""
        if ffmpeg and codec != "h264":
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", src,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-preset", "veryfast", "-movflags", "+faststart", dst],
                           check=True)
            n_conv += 1
        else:
            shutil.copy2(src, dst)
    print(f"videos: {n_conv} transcoded to H.264, {len(data) - n_conv} copied")
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    size = sum(os.path.getsize(os.path.join(dp, fn))
               for dp, _, fns in os.walk(out) for fn in fns)
    print(f"\n{len(data)} clips -> {out}/index.html  ({size / 1e6:.1f} MB total)")
    print("open index.html directly, or `python -m http.server` in that folder "
          "if the browser blocks local video")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
