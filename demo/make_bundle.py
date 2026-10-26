#!/usr/bin/env python
"""Pack the demo into one folder that runs on a laptop (2026-10-12).

The showcase machine will not have this server's directory tree, so the bundle
carries everything: the UI, the collaborator's standalone inference package,
the clips themselves, and the precomputed cache. app.py and inference.py
prefer bundled paths when they exist, so the copied folder needs no edits.

The cache is the part that matters on the day. With it, clicking a clip is
instant; without it, each clip costs ~15 s of CPU and a laptop may be slower.
Build it before packing (python build_cache.py) and the bundle carries the
results; the code path is still real inference, just already run.

    python make_bundle.py                    # -> ./demo_bundle
    python make_bundle.py --out /tmp/demo --zip
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
Z_SH = os.path.dirname(HERE)
PKG_SRC = os.path.join(os.path.dirname(Z_SH), "z-final-sh", "standalone_package")
CODE = ("app.py", "inference.py", "overlay.py", "build_cache.py", "clips.json",
        "thresholds.json")   # per-domain cut-offs; without it the app silently
                             # falls back to a flat 0.5 on any uncached clip

README = """# CCTV 인페인팅 위조 탐지 — 데모

## 실행

```
pip install -r requirements.txt
python app.py
```

브라우저에서 http://127.0.0.1:7860 이 열립니다.
외부에 공유할 임시 링크가 필요하면 `python app.py --share` (72시간 유효).

## 사용법

클립을 고르고 **분석**을 누릅니다. 결과는 네 가지입니다.

- **영상 오버레이** — 모델이 의심하는 영역. Patch-MIL의 4×4 패치 점수라
  대략적인 영역이지 정밀한 마스크가 아닙니다.
- **프레임별 점수 그래프** — 임계값을 넘는 구간이 음영으로 표시됩니다.
- **분기별 기여도** — 다섯 분기 중 어디가 반응했는지.
- **의심 구간 표** — 시작·끝 시각.

같은 장면이 **원본 1개와 조작본 2개**로 들어 있고, 어느 쪽인지는
**정답 공개**를 누르기 전까지 표시되지 않습니다. 분석 결과를 먼저 보고
맞혀볼 수 있습니다.

## 중요

- 모든 클립은 **학습에 사용되지 않은 test 분할**입니다.
- 모델은 **도메인별로 적응**되어 있습니다. 5개 환경(lasiesta, shanghaitech,
  cdnet, streetscene, sbi)에 맞춰 학습·적응된 모델이고, 전혀 다른 환경의
  영상에는 일반화가 약합니다(외부 데이터 기준 macro AUC 약 0.51). **새 현장에
  쓰려면 그 현장 영상 일부를 라벨링해 적응시키는 단계가 필요하며, 그 적응 비용을
  줄이는 것이 이 연구의 목표입니다.** 업로드 기능을 넣지 않은 이유입니다.
- GPU 없이 CPU로 동작합니다. 캐시가 있으면 즉시, 없으면 클립당 약 15초입니다.
"""

RUN_SH = "#!/bin/bash\ncd \"$(dirname \"$0\")\"\npython3 app.py \"$@\"\n"
RUN_BAT = "@echo off\r\ncd /d \"%~dp0\"\r\npython app.py %*\r\npause\r\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "demo_bundle"))
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    import inference

    out = os.path.abspath(args.out)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)

    for name in CODE:
        src = os.path.join(HERE, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(out, name))

    if not os.path.isdir(PKG_SRC):
        sys.exit(f"model package not found: {PKG_SRC}")
    shutil.copytree(PKG_SRC, os.path.join(out, "standalone_package"))
    # The package's own requirements.txt pins "opencv-python==4.10.0", which is
    # not a release -- opencv uses four-component versions (4.10.0.84), so pip
    # refuses the whole file and nothing installs. Rewrite that one line and add
    # the two packages the UI needs on top of the inference package's.
    reqs = []
    with open(os.path.join(PKG_SRC, "requirements.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("opencv-python=="):
                line = "opencv-python>=4.10,<5"
            if line:
                reqs.append(line)
    # gradio 4.x still imports HfFolder, which huggingface_hub removed in 1.0 --
    # an unpinned install picks up 2.x and the app dies on import. Pin the hub.
    # Three pins, all for gradio 4.x's ageing client:
    #   huggingface_hub   -- 1.0 removed HfFolder, which gradio still imports
    #   fastapi           -- 0.113+ emits `additionalProperties: true`, and
    #                        gradio_client's schema walker assumes a dict, so
    #                        every page load 500s and launch then reports
    #                        "localhost is not accessible"
    reqs += ["gradio>=3.40,<5", "huggingface_hub<1.0", "fastapi<0.113",
             "matplotlib>=3.5"]
    with open(os.path.join(out, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(reqs) + "\n")

    with open(os.path.join(HERE, "clips.json"), encoding="utf-8") as f:
        clips = json.load(f)["clips"]
    vids = os.path.join(out, "videos")
    os.makedirs(vids)
    # Six of the ten dataset clips are mp4v, which no browser will decode --
    # the page loads and the video pane just stays blank. Gradio serves these
    # to a browser too, so they get the same H.264 pass the static demo does.
    from build_html_demo import _find_ffmpeg
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        sys.exit("no ffmpeg found; cannot transcode clips to H.264.")
    missing, n_conv = [], 0
    for c in clips:
        try:
            src = inference.resolve_video(c["id"])
        except FileNotFoundError:
            missing.append(c["id"]); continue
        dst = os.path.join(vids, f"{c['id']}.mp4")
        info = subprocess.run([ffmpeg, "-i", src], capture_output=True,
                              text=True).stderr
        line = [l for l in info.splitlines() if "Video:" in l]
        codec = line[0].split("Video: ")[1].split()[0] if line else ""
        if codec == "h264":
            shutil.copy2(src, dst)
        else:
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", src,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-preset", "veryfast", "-movflags", "+faststart",
                            dst], check=True)
            n_conv += 1

    cache_src = os.path.join(HERE, "cache")
    n_cached = 0
    if os.path.isdir(cache_src):
        shutil.copytree(cache_src, os.path.join(out, "cache"))
        n_cached = len([f for f in os.listdir(cache_src) if f.endswith(".json")])

    # The static page is the one actually used at the showcase: it needs no
    # python, no install and no network, so it has to travel with the bundle
    # rather than being rebuilt on the day.
    html_src = os.path.join(HERE, "html_demo")
    n_html = 0
    if os.path.isdir(html_src):
        shutil.copytree(html_src, os.path.join(out, "html_demo"))
        n_html = len([f for _, _, fns in os.walk(html_src)
                       for f in fns if f.endswith(".mp4")])

    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)
    with open(os.path.join(out, "run.sh"), "w") as f:
        f.write(RUN_SH)
    os.chmod(os.path.join(out, "run.sh"), 0o755)
    with open(os.path.join(out, "run.bat"), "w", newline="") as f:
        f.write(RUN_BAT)

    total = sum(os.path.getsize(os.path.join(dp, fn))
                for dp, _, fns in os.walk(out) for fn in fns)
    print(f"bundle -> {out}")
    print(f"  clips     : {len(clips) - len(missing)}/{len(clips)}"
          + (f"  MISSING {missing}" if missing else ""))
    print(f"  cached    : {n_cached} clips  "
          f"({'instant at showcase' if n_cached else 'WILL COMPUTE ON DEMAND'})")
    print(f"  videos    : {len(clips) - len(missing)} ({n_conv} transcoded to H.264)")
    print(f"  html_demo : {n_html} videos"
          if n_html else "  html_demo : MISSING -- run build_html_demo.py")
    print(f"  total size: {total / 1e6:.1f} MB")

    if args.zip:
        base = shutil.make_archive(out, "zip", root_dir=os.path.dirname(out),
                                   base_dir=os.path.basename(out))
        print(f"  zip       : {base} ({os.path.getsize(base) / 1e6:.1f} MB)")
    print("ALL_DONE")


if __name__ == "__main__":
    main()
