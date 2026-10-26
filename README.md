# NOC_CCTV-Detection

CCTV 영상에서 **객체 제거 인페인팅 조작**을 탐지하는 5-브랜치 포렌식 융합 모델과, 졸업작품 시연용 데모입니다.

논문 *A Multi-Cue Fusion Framework for Cross-Domain Inpainting Forgery Detection in CCTV Videos* (Songhye Kim, Hagyeong Kim, Cheolsoo Park, Jihyeon Ryu / 광운대학교) 의 코드입니다.

- 영상 1개를 넣으면 프레임별 조작 의심 점수와 조작 구간을 돌려줍니다.
- 5가지 서로 다른 포렌식 증거를 각각 학습해 융합합니다.
- GPU 없이 CPU만으로 동작합니다 (영상 1개당 약 8초).

---

# 하는 일

객체 제거 인페인팅은 영상에서 사람이나 물체를 지우고 배경을 그럴듯하게 복원합니다. 눈으로는 경계가 보이지 않는 경우가 많고, 조작 흔적이 항상 경계선 형태로 남지도 않습니다. 그래서 한 가지 단서만 보지 않고 5가지를 각각 따로 학습한 뒤 합칩니다.

## 5개 브랜치

| 브랜치 | 입력 | 모델 |
| --- | --- | --- |
| Statistical | 프레임 기술자 17개 + 클립 통계 30개 (47차원) | LightGBM |
| Compression | PRNU/블록 특징 6개 + 클립 통계 30개 (36차원) | LightGBM |
| Residual | 포렌식 맵 4장, 4×32×32 텐서 | CNN + 1D-TCN |
| Local patch | 원해상도 맵 크롭, 패치당 4×16×16 | 공유 CNN + MIL |
| Temporal frequency | 프레임별 스펙트럼 + 활동량 입력 | MLP + MIL |

## 융합 방식

각 브랜치의 점수는 **source-CDF**로 먼저 보정됩니다. 브랜치마다 점수 범위가 전혀 다르기 때문에, 그대로 평균하면 특정 브랜치가 결과를 지배합니다.

```text
브랜치 원점수 → source-CDF 백분위 보정 → 5개 동일 가중 평균 → 인과적 평활
```

- 가중치는 학습하지 않고 **균등 평균**입니다. 학습된 융합 가중치를 실험해봤지만 적은 적응 데이터에 과적합했습니다.
- 평활은 **인과적(causal)** 입니다. 한 프레임의 점수는 과거 프레임만 사용하며 미래 프레임을 보지 않습니다.
- Ablation에서 source-CDF를 빼면 macro AUC가 0.1411 떨어집니다. 브랜치 하나를 빼는 것(최대 0.0300)보다 훨씬 큽니다. 이 프로젝트에서 가장 중요한 구성 요소는 보정 단계입니다.

---

# 시작하기

## 1. 요구 사항

### 공통 환경

- Python 3.9 이상
- GPU 불필요 (CPU 전용)
- `ffmpeg` 바이너리는 `imageio-ffmpeg` 패키지가 자동으로 받아오므로 따로 설치하지 않아도 됩니다.

### 추가로 필요한 것

- **학습된 체크포인트** — 이 저장소에 없습니다. 아래 *체크포인트와 영상* 참고.
- **영상** — 데모를 돌리려면 원본 데이터셋이 필요합니다. 아래 *체크포인트와 영상* 참고.

## 2. 저장소 내려받기

```bash
git clone https://github.com/Hagyeong13/NOC_CCTV-Detection.git
cd NOC_CCTV-Detection
```

## 3. 가상환경 만들기

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

## 4. 패키지 설치

```bash
pip install --upgrade pip
pip install -r model/requirements.txt
```

---

# 실행 방법

## 단일 영상 실행

```bash
cd model
python3 run_inference.py --video /path/to/clip.mp4 --domain sbi
```

출력 예시:

```text
=== result ===
video: /path/to/clip.mp4
domain model: sbi (adapted/DA)
frames: 80  mean_score=0.8180  max_score=0.9080
frames >= threshold(0.5): 80/80 (100.0%)
elapsed: 8.4s
```

## 도메인 지정

`--domain`은 5개 중 하나를 고릅니다. 모델이 도메인별로 따로 있기 때문에 필수입니다.

```text
lasiesta  shanghaitech  cdnet  streetscene  sbi
```

## 적응 전 모델로 실행

```bash
python3 run_inference.py --video clip.mp4 --domain sbi --zero-shot
```

## 임계값 바꾸기

```bash
python3 run_inference.py --video clip.mp4 --domain sbi --threshold 0.4381
```

기본값은 `0.5`인데, **도메인별 보정값을 쓰는 것을 권합니다.** 아래 *임계값* 참고.

## 프레임별 점수를 CSV로 저장

```bash
python3 run_inference.py --video clip.mp4 --domain sbi --out-csv scores.csv
```

## 체크포인트 폴더 지정

```bash
python3 run_inference.py --video clip.mp4 --domain sbi --ckpt-root /path/to/checkpoints
```

## 중간 파일 남기기

```bash
python3 run_inference.py --video clip.mp4 --domain sbi --keep-work-dir
```

## 실행 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--video` | (필수) | 입력 영상 경로 |
| `--domain` | (필수) | 5개 도메인 중 하나 |
| `--ckpt-root` | 패키지 내 `checkpoints/` | 체크포인트 폴더 |
| `--zero-shot` | 꺼짐 | 적응 전(no-DA) 모델 사용 |
| `--threshold` | `0.5` | 조작 판정 임계값 |
| `--out-csv` | 없음 | 프레임별 점수 CSV 저장 경로 |
| `--keep-work-dir` | 꺼짐 | 중간 파일 유지 |

---

# 출력 CSV 컬럼

`--out-csv`로 저장하면 3개 컬럼이 나옵니다.

### `frame`

프레임 번호입니다. 1부터 시작합니다.

### `score`

해당 프레임의 조작 의심 점수입니다. 0에 가까우면 정상, 1에 가까우면 조작입니다. 5개 브랜치를 보정·융합하고 인과적 평활까지 끝낸 최종 값입니다.

### `flagged`

`score >= --threshold`이면 1, 아니면 0입니다.

이 CSV는 **최종 진위 판정이 아니라 프레임별 점수**입니다. 클립 단위 판정은 아래 *판정 규칙*을 따릅니다.

---

# 임계값

플랫 `0.5`는 적절하지 않습니다. 도메인마다 점수 분포가 달라서, 같은 임계값이 도메인에 따라 전혀 다른 오탐율을 냅니다.

`demo/thresholds.json`에 in-domain adapt split(seed 0)에서 **오탐율 5%** 기준으로 보정한 값이 들어 있습니다.

| 도메인 | 임계값 | 실제 오탐율 | 재현율 |
| --- | --- | --- | --- |
| cdnet | 0.7264 | 0.0500 | 0.6486 |
| lasiesta | 0.1620 | 0.0458 | 0.9915 |
| sbi | 0.4381 | 0.0495 | 0.8768 |
| shanghaitech | 0.5164 | 0.0413 | 0.8473 |
| streetscene | 0.5385 | 0.0391 | 0.7263 |

0.16에서 0.73까지 벌어집니다. 도메인 보정 없이 한 값을 쓰면 안 되는 이유가 이 표입니다.

직접 다시 보정하려면 다음과 같이 실행합니다.

```bash
cd demo
python3 calibrate_thresholds.py --fpr 0.05 --seed 0 --out thresholds.json
```

---

# 판정 규칙

프레임 하나가 임계값을 넘었다고 조작으로 판정하지 않습니다.

- 프레임 단위 오탐율 10%는, 200프레임 정상 클립에서 **약 20프레임이 임계값을 넘는다**는 뜻입니다.
- 그래서 **연속 24프레임(24fps에서 1초) 이상** 넘어야 하나의 조작 구간으로 보고합니다.
- 프레임 단위 오탐 예산과 클립 단위 판정 규칙은 서로 다른 것입니다.

---

# 데모

## 데모 설계

- 평가자가 고정된 메뉴에서 클립을 고릅니다. **업로드 기능은 의도적으로 없습니다.** 모델이 도메인별이라 임의 영상을 받으면 실제보다 좋은 인상을 주게 됩니다.
- 모든 클립은 held-out 테스트 split에서 나옵니다. "외운 거 아니냐"에 답할 수 있습니다.
- 각 장면이 원본과 조작본으로 같은 조건에서 나오고, **라벨은 '정답 보기'를 누를 때까지 숨겨집니다.** 조작본을 잡는 것과 원본을 건드리지 않는 것을 둘 다 봐야 하기 때문입니다.
- 실제 추론 결과를 클립별로 캐시해두므로 발표 중에는 즉시 뜹니다. 캐시가 없고 모델도 없으면 점수를 지어내지 않고 없다고 표시합니다.

## 데모 만들기

```bash
cd demo
python3 build_cache.py          # 실제 추론 후 클립별 캐시
python3 build_html_demo.py      # -> html_demo/index.html (단일 파일, 설치 불필요)
python3 app.py                  # Gradio UI, 127.0.0.1:7860
```

## HTML 데모 열기

`html_demo/index.html`을 브라우저로 바로 열면 됩니다. 브라우저가 로컬 영상 재생을 막는 경우에는 해당 폴더에서 간단한 서버를 띄웁니다.

```bash
cd html_demo
python3 -m http.server 8000
```

그다음 `http://localhost:8000`으로 접속합니다.

## Gradio 앱 옵션

```bash
python3 app.py --port 7861              # 포트 변경
python3 app.py --host 0.0.0.0           # 다른 기기에서 접속 허용
python3 app.py --share                  # 공개 링크 생성
```

---

# 폴더 설명

```text
NOC_CCTV-Detection/
├── model/
│   ├── run_inference.py       # 단일 영상 추론 진입점
│   ├── requirements.txt
│   └── vendor/                # 5-브랜치 구현 (무수정 포함)
└── demo/
    ├── app.py                 # Gradio UI
    ├── build_html_demo.py     # 단일 파일 HTML 데모 생성
    ├── inference.py           # 데모용 추론 래퍼
    ├── overlay.py             # 영상에 점수 바 합성
    ├── clips.json             # 데모 클립 목록 + 정답 구간
    └── thresholds.json        # 도메인별 보정 임계값
```

## `model/` — 추론

### `run_inference.py`

영상 1개를 받아 프레임별 점수와 요약을 출력합니다. 특징 추출부터 융합까지 전 과정을 수행합니다.

### `vendor/`

5-브랜치 모델 구현입니다. 공동저자가 작성했고 **논문 버전과의 재현성을 위해 수정하지 않고 그대로 포함**했습니다. 일부 모듈에 개발 당시의 절대 경로가 남아 있지만 학습·특징추출 경로에서만 읽히므로 `run_inference.py` 실행에는 영향이 없습니다. 자세한 내용은 `model/README_VENDOR.md`를 참고하세요.

## `demo/` — 시연

### `build_cache.py`

데모 클립 전체를 미리 추론해 캐시합니다. `--force`로 다시 계산하고 `--only`로 일부만 처리합니다.

### `build_html_demo.py`

캐시를 읽어 **설치가 필요 없는 단일 HTML 파일**을 만듭니다. 영상은 브라우저가 재생할 수 있도록 H.264로 변환해 넣습니다.

### `calibrate_thresholds.py`

도메인별 임계값을 보정합니다. `--fpr`로 목표 오탐율을 지정합니다.

### `select_demo_clips.py` / `measure_visual_subtlety.py`

데모에 넣을 클립을 고릅니다. 모델이 잘 맞히는 순서로만 고르면 눈에 뻔히 보이는 조작만 모이기 때문에, PSNR·SSIM·차이 면적으로 **시각적 미묘함을 따로 측정**해서 함께 봅니다.

### `occlusion_probe.py`

어느 영역이 점수에 기여했는지 보는 탐침입니다. 한 영역을 가리고 전체를 다시 추론해 점수가 얼마나 떨어지는지 측정합니다.

```bash
python3 occlusion_probe.py --clip <클립ID> --grid 4
```

---

# 성능

5개 데이터셋(LASIESTA, ShanghaiTech, CDnet2014, StreetScene, SBI)의 201개 장면에서, 장면마다 원본 1개와 조작본 2개(ProPainter, MiniMax-Remover)를 만들었습니다. 총 603개 변형, 130,625 라벨 프레임, 조작 프레임 비율 44.3%입니다.

Leave-one-domain-out, 6 seed, 프레임 단위 ROC-AUC:

| 프로토콜 | Macro AUC |
| --- | --- |
| LODO (타깃 적응 포함) | 0.7920 ± 0.0082 |
| LODO (적응 없음, 동일 조건 대조군) | 0.6837 ± 0.0086 |
| Pooled 70/15/15 | 0.8517 ± 0.0571 |
| In-domain | 0.8898 ± 0.0270 |

- 분할은 **장면 그룹 단위**입니다. 같은 장면의 원본과 조작본이 학습·적응·평가에 나뉘어 들어가지 않습니다.
- LODO 수치는 타깃 도메인 장면 그룹의 약 20%를 라벨 있는 적응 데이터로 받고 나머지로 평가한 결과입니다.
- 비교 대상인 외부 모델들은 타깃 라벨을 받지 않습니다. 따라서 이 표는 **라벨 예산이 서로 다른 상태의 시스템 단위 비교**이고, 구조만의 비교가 아닙니다. 라벨 예산을 맞춘 비교는 진행 중입니다.

---

# 체크포인트와 영상

둘 다 이 저장소에 **의도적으로 포함하지 않았습니다.**

## 체크포인트

- 도메인별 학습 모델 5개(약 27MB)는 공동저자의 작업물이라 재배포하지 않습니다.
- 필요하면 저자에게 요청하세요.

## 영상

- 데모 클립은 CDnet2014, LASIESTA, SBI, ShanghaiTech, StreetScene에서 파생된 것입니다.
- 이 데이터셋들의 이용 약관이 재배포를 허용하지 않습니다.
- 각 데이터셋 배포처에서 직접 받은 뒤 로컬에서 클립을 다시 만들어야 합니다.

둘이 없어도 스크립트는 실행되며, 점수를 지어내지 않고 **무엇이 없는지 알려줍니다.**

---

# 현재 한계

- 배포된 도메인별 모델은 **in-domain 모델**입니다. 각자 자기 도메인 70%로 학습하고 15%로 few-shot 적응한 것이며, 논문의 cross-domain 수치를 낸 LODO 모델이 아닙니다.
- 완전히 새로운 미지의 도메인에서는 macro AUC가 **0.51 수준**으로 거의 랜덤까지 떨어집니다. "5개 도메인과 비슷한 CCTV 장면"용으로 생각해야 합니다.
- 데모에 업로드 기능이 없는 이유가 이것입니다.
- 출력은 프레임별 점수이며 최종 진위 판정이 아닙니다. 클립 판정에는 연속 24프레임 규칙이 필요합니다.
- 임계값은 도메인별로 보정해야 합니다. 플랫 `0.5`는 도메인에 따라 오탐율이 크게 달라집니다.
- 융합 가중치는 학습하지 않습니다. 적은 적응 데이터에서 과적합했기 때문입니다.

---

# 히트맵을 넣지 않은 이유

데모에는 히트맵 오버레이가 없습니다. 처음에는 Patch-MIL 맵을 영상에 올렸는데 두 가지 문제가 있었습니다.

- 4×4 격자는 프레임의 1/16 단위까지만 짚습니다. 화면에 띄우면 모델이 뒷받침할 수 없는 **"어디"에 대한 주장**으로 읽힙니다.
- Patch-MIL 점수는 패치들 사이의 **상대 순위**입니다. 그래서 조작이 없는 영상에서도 움직이는 사람이 있는 패치가 가장 높은 점수를 받습니다. 실제로 원본 클립에서 빨간 영역이 사람을 따라다니는데 판정은 "정상"으로 나왔습니다.

모델이 뒷받침할 수 있는 것은 **"언제"** 이고, 그건 점수 곡선과 구간 표가 전달합니다. "어디"를 보려면 `occlusion_probe.py`를 쓰는 것이 정직한 방법입니다. 한 영역을 가리고 다시 추론해서 점수가 떨어지는 정도를 측정하므로, 모델의 실제 동작에서 나온 값입니다.

---

# 빠른 실행 요약

```bash
git clone https://github.com/Hagyeong13/NOC_CCTV-Detection.git
cd NOC_CCTV-Detection

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r model/requirements.txt

cd model
python3 run_inference.py --video /path/to/clip.mp4 --domain sbi --threshold 0.4381
```

체크포인트가 필요합니다. 없으면 스크립트가 알려줍니다.
