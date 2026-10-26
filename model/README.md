# CCTV Inpainting-Detection — Standalone Inference Package

완전 독립 패키지입니다. 서버의 project4 경로를 전혀 참조하지 않고, 이 폴더
하나만 있으면 어느 컴퓨터에서든 바로 돌아갑니다 — 격리된 디렉토리에서 깨끗한
환경(`env -i`, 외부 PYTHONPATH 없음)으로 실행해서 원본 서버 버전과 동일한
결과가 나오는 것까지 확인했습니다.

5개 도메인(lasiesta, shanghaitech, cdnet, streetscene, sbi) 각각 자기 도메인
데이터의 70%로 학습(pretrain_70) 후 자기 도메인 15% adapt subset으로 few-shot
적응(+adapt)까지 끝낸 모델입니다. seed=0 기준 단일 체크포인트입니다.

**이 모델의 성격**: 각 도메인 모델은 *그 도메인 전용*입니다. 완전히 새로운 미지의
도메인(5개 도메인 어디에도 안 속하는 영상)에는 일반화가 약합니다 — 같은 실험에서
IITB-Corridor(외부 데이터)로 평가했을 때 macro AUC가 0.51~0.52로 거의 랜덤 수준까지
떨어지는 걸 확인했습니다. "5개 도메인과 비슷한 CCTV 장면"에 쓰는 용도로 생각하시면
됩니다.

## 설치

```bash
pip install -r requirements.txt
```

`ffmpeg` 바이너리 자체는 `imageio-ffmpeg` 패키지가 자동으로 받아오므로 시스템에
따로 설치할 필요 없습니다. GPU도 필요 없습니다 (CPU 전용, 영상 1개당 약 8초).

## 사용법

```bash
python3 run_inference.py --video /path/to/new_video.mp4 --domain sbi
```

출력 예시:
```
=== result ===
video: /path/to/new_video.mp4
domain model: sbi (adapted/DA)
frames: 80  mean_score=0.8180  max_score=0.9080
frames >= threshold(0.5): 80/80 (100.0%)
elapsed: 8.4s
```

옵션:
- `--domain`: `lasiesta`/`shanghaitech`/`cdnet`/`streetscene`/`sbi` 중 영상
  환경(실내/실외, 카메라 고정 여부 등)이 가장 비슷한 것을 고르면 됩니다. 확실하지
  않으면 5개 다 돌려서 비교해도 됩니다.
- `--out-csv result.csv`: 프레임별 점수를 CSV로 저장
- `--zero-shot`: adapted 대신 zero-shot(적응 안 한) 체크포인트 사용
- `--threshold 0.5`: "조작 의심" 판정 기준점 (기본 0.5)

## 성능 (own-test, 해당 도메인 자신의 15% held-out 테스트셋 기준)

| domain | pretrain_70 (zero-shot) AUC | +adapt (DA) AUC |
|---|---|---|
| lasiesta | 0.8935 | 0.9209 |
| shanghaitech | 0.7082 | 0.6715 |
| cdnet | 0.8361 | 0.7920 |
| streetscene | 0.9042 | 0.8513 |
| sbi | 0.9457 | 0.8021 |

도메인에 따라 adapt가 오히려 AUC를 깎는 경우도 있습니다(같은 도메인 내 적응이라
실제 distribution shift가 없을 때 생기는 현상). 기본 배포 모델은 `*_adapted.*`
(DA 포함 버전)으로 고정했지만, `--zero-shot`로 바로 비교해볼 수 있습니다.

## 동작 원리 (요약)

1. 입력 비디오를 표준 코덱(24fps, libx264, CRF 23, yuv420p)으로 재인코딩
2. 5개 evidence branch를 각각 돌림:
   - LGBM(통계, 17 feature) / LGBM(압축·PRNU, 6 feature) / CNN(residual+TCN)
   - Patch-MIL(공간 패치 단위) / TempFreq(시간-주파수 FFT 기반)
3. 각 branch 점수를 **source-CDF 캘리브레이션**(학습 도메인 기준 분포에 대한
   percentile)으로 정규화 후 단순평균
4. **Causal K=32 rolling mean**으로 시간축 스무딩 (미래 프레임을 보지 않음 —
   실시간 스트리밍에도 적용 가능한 구조)

## 폴더 구조

```
standalone_package/
  README.md
  requirements.txt
  run_inference.py        (진입점, vendor/ 안의 모듈만 사용)
  vendor/                 (feature 추출 + 모델 아키텍처 코드, 전부 이 안에 포함됨)
  checkpoints/
    all_domains_summary.json
    <domain>/
      lgbm_adapted.txt / lgbm_zeroshot.txt           (통계 branch)
      compression_adapted.txt / compression_zeroshot.txt   (PRNU/압축 branch)
      cnn_adapted.pt / cnn_zeroshot.pt               (residual CNN+TCN)
      patch_adapted.pt / patch_zeroshot.pt           (Patch-MIL)
      tempfreq_adapted.pt / tempfreq_zeroshot.pt     (temporal-frequency)
      fusion_reference.npz                            (source-CDF 캘리브레이션 기준)
      metadata.json
```

## 검증

기존 학습 데이터의 pristine/manipulated(propainter) 한 쌍으로 end-to-end
테스트: pristine mean=0.5663, manipulated mean=0.8180 — 방향성 정상. 또한 이
비디오 기반 추출 경로가 기존 CSV 기반 공식 학습 파이프라인과 거의 동일한 점수
(0.5663 vs 0.5684, 같은 케이스)를 내는 것도 확인해서, 벤더링 과정에서 재구현
버그가 들어가지 않았다는 것도 교차 검증했습니다.
