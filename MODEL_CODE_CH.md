# 작업 지침: 베이스라인 코드 고치기와 코랩 실행 도구 만들기

이 문서만 읽고 일할 수 있게 썼다. 이번 작업은 **코드만 고친다. 실제 학습은 하지 않는다.** 학습은 사용자가 나중에 구글 코랩(Colab Pro+, GPU)에서 혼자 한다. 그때는 Claude Code가 도와줄 수 없으므로, 사용자가 **설정 몇 개만 바꿔서** 모든 실험을 돌릴 수 있는 도구를 만드는 것이 이번 작업의 절반이다.

원칙은 지난 작업들과 같다. 필수 확인만 하고 바로 만든다. 애매하면 질문하지 말고 기본값으로 정한 뒤 `CHANGES.md`에 한 줄 적는다.

## 1. 배경 (짧게)

STARSS23로 세 가지 데이터셋을 만들었다. 각각 `README_dataset.md`, `BUILD_REPORT.md`, `NOTES.md`에 설명이 있다.

- `main20`: 20초 클립, 카메라가 돈다. 챌린지 본 데이터이고 공개한다.
- `ctrl5`: main20의 창을 5초씩 4등분하고 카메라를 고정했다. 내부 대조용이다.
- `fix20`: main20의 val과 test 창을 20초 그대로, 카메라를 고정했다. 내부용이고 train은 없다.

라벨 형식은 `frame,class,source,azimuth,distance,onscreen`(헤더 있음)이다. 방위각은 카메라 기준 −180~180이고 **접혀 있지 않다.** 오디오는 스테레오 24kHz, 영상은 640×360 **10fps**다.

챌린지의 핵심 숫자는 **"영상 덕분에 오른 점수" = (영상+소리 모델) − (소리 모델)**이다. 이것을 main20과 ctrl5에서 각각 구해 비교한다.

## 2. 작업 위치

- 원본 베이스라인은 `baseline/DCASE2025_seld_baseline/`에 있다. **원본은 고치지 말고** `baseline/seld_challenge/`로 복사해서 거기서 고친다.
- 데이터 파일은 고치지 않는다.
- 아래의 줄 번호는 지난 리뷰 기준이라 다를 수 있다. 직접 읽고 찾아라.

## 3. 필수 수정 (안 고치면 학습이 안 되거나 점수가 틀린다)

1. **영상 프레임 간격** (`utils.py` `load_video`, 115~116행 근처): fps를 30으로 고정하고 3장마다 1장을 쓰는 코드를, 실제 fps를 읽어 `간격 = round(fps / 10)`으로 계산하도록 바꾼다(우리 영상은 1이 된다). 특징을 뽑은 뒤 영상 프레임 수가 라벨 프레임 수(200 또는 50)와 다르면 즉시 에러를 내게 한다.
2. **방위각 오차 계산** (`utils.py` `fold_az_angle`, `least_distance_between_gt_pred`): 각도 차이를 원형 차이 `|((a − b + 180) mod 360) − 180|`로 계산한다. 기존처럼 접는 방식은 옵션으로 남긴다. 채점은 항상 두 방식으로 모두 한다(5절).
3. **폴더와 이름 규칙**: `parameters.py`, `extract_features.py`, `data_generator.py`, `evaluate.py`, `inference.py`에서 fold 이름(`fold1`, `fold3`, `fold4`)과 `stereo_dev/dev-*` 같은 경로를 우리 구조 `{dataset}/{split}/{audio,video,labels}/`와 split 이름 `train`, `val`, `test`로 바꾼다. test 라벨은 `{dataset}/private/test_labels/`에 있다.
4. **라벨 길이**: `label_sequence_length`를 데이터셋에 따라 자동으로 정한다(main20과 fix20은 200, ctrl5는 50).
5. **정답 폴더 읽기** (`metrics.py` 219~225행 근처): 하위 폴더를 가정하는 코드를, csv가 바로 들어 있는 폴더(`val/labels/`)를 읽게 바꾼다.
6. **모델 고르기** (`main.py` 109, 121, 138, 165행 근처): 매 에폭의 점수를 test가 아니라 **val**로 계산하고, 그 점수로 가장 좋은 모델을 고른다.
7. **val에 없는 클래스** (`metrics.py` 87행 근처): 정답이 하나도 없는 클래스는 평균 F에서 뺄 수 있게 옵션을 둔다(기본은 뺀다). val에는 9번(악기)과 11번(종)이 없다.
8. **배치 크기**: 설정으로 바꿀 수 있게 한다. 기본값은 `auto`이고, `12800 ÷ 라벨 프레임 수`로 정한다(main20 64, ctrl5 256). 이렇게 하면 한 배치에 들어가는 프레임 수가 같고, 에폭당 학습 횟수도 두 데이터셋이 거의 같다(2999/64 ≈ 11996/256 ≈ 47).
9. **채점 프레임 수** (`metrics.py` 223행 근처, `nb_ref_frames = max(frame)`): 정답 파일의 마지막 프레임 대신 **클립 길이**(200 또는 50)를 쓴다. 지금은 정답이 없는 파일과 마지막 소리 이후 구간의 예측을 오검출로 세지 않는다. 이 문제는 클립이 짧고 많은 ctrl5의 점수를 부풀린다. 같은 수정을 채점 스크립트(5절)에도 넣는다.
10. **기타**: `nb_workers`를 설정으로 뺀다(기본 2). 기존 체크포인트는 접힌 라벨로 학습된 것이므로 쓰지 않는다(처음부터 학습).

## 4. 모델 설정 (옵션으로 구현)

| 옵션 | 값 | 뜻 |
|---|---|---|
| `modality` | `audio` / `audio_visual` | 소리만 쓰는 원래 모델 / 영상+소리 원래 모델 |
| `video_input` | `real` / `zeros` | 학습과 평가 모두에서 영상 특징을 그대로 쓰거나 0으로 채운다. `zeros`는 영상+소리 모델과 **같은 구조**에서 영상만 없는 대조군이다. |
| `video_crop` | `center` / `full` | `center`는 원래 전처리 그대로다. `full`은 화면을 자르지 않고 높이 224, 너비 448로 줄여 ResNet에 넣는다(특징 지도 7×14, 채널 평균 후 98차원). 특징 차원은 설정에서 읽어 `Linear` 입력 크기를 맞춘다. |
| `av_time_window` | `none` / 정수 k | `none`은 원래대로다(마스크 없음). k이면 디코더의 교차 주의에 `memory_mask`를 넣어, 오디오 프레임 t가 영상 프레임 t−k~t+k만 보게 한다. PyTorch bool 마스크는 True가 "못 봄"이다. 오디오 프레임 수와 영상 프레임 수가 같은지(200 또는 50) 확인하는 assert를 넣는다. |

**프리셋**: 아래 4개를 만든다. 사용자는 보통 이것만 고른다.

- `B1`: `modality=audio`
- `B2`: `modality=audio_visual`, `video_input=real`, `video_crop=center`, `av_time_window=none`
- `B2-0`: B2에서 `video_input=zeros`
- `B3`: B2에서 `video_crop=full`, `av_time_window=2`

카메라 자세는 어떤 프리셋에서도 모델 입력에 넣지 않는다.

## 5. 채점

`score.py` 하나로 학습 중 val 채점, 최종 val 채점, test 채점(주최자용), 챌린지 리더보드 채점을 모두 한다.

- 입력은 예측 csv 폴더, 정답 csv 폴더, 클립 길이다. 예측 형식은 베이스라인의 `write_to_dcase_output_format` 출력과 같다.
- **항상 두 방식으로 모두 계산한다.**
  - 확장 점수: 방위각을 접지 않는다. 순위와 모델 고르기에 쓴다.
  - 기본 점수: 정답과 예측을 둘 다 앞쪽으로 접는다. DCASE 방식이다.
- 출력 항목은 F(20°, 상대 거리 1.0), DOA 오차, 상대 거리 오차, onscreen 정확도, 클래스별 F다.
- 소리 모델(B1)은 onscreen을 내지 않는다. onscreen이 들어간 점수는 영상+소리 모델에만 계산하고, B1에서는 `NA`로 적는다. **"영상 덕분에 오른 점수"는 onscreen이 빠진 F로만 계산한다.**
- `--exclude_list` 옵션으로 특정 클립을 빼고 채점할 수 있게 한다. fix20에서 빠진 `val_00088`과, 그 창에 해당하는 main20 클립과 ctrl5 조각 4개를 빼고 비교할 때 쓴다(ctrl5의 대응은 `ctrl5/private/parent_map.csv`에 있다).
- 결과는 json(전체와 클래스별)과 한 줄 요약으로 낸다.

## 6. 코랩 실행 도구 (가장 중요)

사용자는 코랩에서 **노트북 셀을 위에서부터 실행하고, 설정 칸만 바꾼다.** 아래를 만든다.

### 6.1 설정 파일

`configs/base.yaml`에 모든 설정과 기본값을 둔다. 각 줄에 한국어 주석으로 뜻과 가능한 값을 적는다. 적어도 아래 항목이 있어야 한다.

```yaml
drive_root:        # 드라이브의 프로젝트 폴더 (결과, 체크포인트, 특징 저장 위치)
data_root: /content/data   # 코랩 로컬로 풀어 둘 데이터 위치
dataset: main20    # main20 | ctrl5 | fix20
preset: B2         # B1 | B2 | B2-0 | B3 (아래 개별 옵션이 비어 있으면 프리셋 값을 쓴다)
modality:
video_input:
video_crop:
av_time_window:
seed: 1
epochs: 200
lr: 0.001
batch_size: auto
num_workers: 2
select_metric: F_ext          # F_ext | F_basic
exclude_absent_classes: true
eval_video: real              # real | zeros | shuffle  (평가 때만, 아래 6.3)
resume: true
```

`run.py`는 `--set key=value`로 어떤 값이든 덮어쓸 수 있게 한다.

### 6.2 실행 명령

`run.py <단계> --preset B2 --dataset main20 --set seed=2` 형태로 쓴다. 단계는 다음과 같다.

| 단계 | 하는 일 |
|---|---|
| `prepare` | 드라이브의 zip을 `data_root`에 푼다. 이미 풀려 있으면 건너뛴다. |
| `features` | 오디오와 영상 특징을 뽑아 **드라이브에 캐시**한다. 이미 있으면 건너뛴다. 오디오 특징은 프리셋과 무관하게 한 번만 뽑고, 영상 특징은 `video_crop`별로 따로 저장한다. |
| `train` | 학습한다. 매 에폭 마지막 체크포인트와 가장 좋은 체크포인트를 드라이브에 저장한다. `resume=true`이면 끊긴 곳부터 이어 한다. 에폭별 손실과 val 점수를 csv로 남긴다. |
| `eval` | 가장 좋은 체크포인트로 val을 채점한다. |
| `swaptest` | 같은 체크포인트로 `eval_video=shuffle`(영상을 다른 클립 것으로 바꿈, 시드 고정, 어떤 클립도 자기 영상을 받지 않게)과 `eval_video=zeros`로 val을 채점한다. 영상+소리 모델에서만 동작한다. |
| `predict` | test 예측 csv를 만들고 제출용 zip으로 묶는다. |
| `score` | 주최자용. `private_test` 정답으로 test 예측을 채점한다. |
| `summary` | 모든 실행 결과를 `results/summary.csv` 한 표로 모은다. 같은 데이터셋·시드에서 B2−B1, B2−(B2-0), B3−B1의 F 차이를 확장·기본 두 방식으로 계산해 `results/gaps.csv`로 낸다. 시드가 여럿이면 평균과 표준편차도 낸다. |

**실행 이름**은 `{dataset}_{preset}_s{seed}`로 자동으로 정한다. 프리셋과 다른 옵션을 덮어썼으면 뒤에 짧게 붙인다. 각 실행 폴더에 그때 쓴 설정 전체(`config_used.yaml`)를 저장한다.

### 6.3 노트북

`colab_run.ipynb`를 만든다. 셀 순서는 다음과 같다.

1. 드라이브 마운트, 프로젝트 폴더 경로 지정, 패키지 설치
2. **설정 칸**: 코랩 폼(`#@param`)으로 dataset, preset, seed, epochs, batch_size, lr, eval_video를 드롭다운이나 입력칸으로 둔다.
3. `prepare`
4. `features`
5. `train`
6. `eval` → `swaptest`
7. `predict`
8. `summary` 결과 표 보기

각 셀 위에 한두 문장으로 무엇을 하는 셀인지, 오래 걸리는지 적는다.

### 6.4 데이터 옮기기

- 드라이브에서 작은 파일 수천 개를 직접 읽으면 매우 느리다. 그래서 코랩에서는 zip을 로컬로 풀어서 쓴다.
- main20에는 공개 zip이 이미 있다. ctrl5와 fix20에는 zip이 없으므로, 이번 작업에서 **내부 전송용 zip**을 만든다. 이름은 `internal_ctrl5_{split}.zip`, `internal_fix20_{split}.zip`이고, test 라벨은 `internal_{dataset}_private.zip`에 따로 넣는다.
- 이 zip들은 공개 폴더가 아닌 `internal/` 폴더에 둔다. 공개 zip에는 손대지 않는다.

### 6.5 사용 설명서

`README_baseline.md`를 쉬운 한국어로 쓴다. 들어갈 내용은 다음과 같다.

- 코랩에서 처음부터 끝까지 돌리는 순서
- 각 설정의 뜻
- 권장 실험 순서: main20의 B1, B2 → ctrl5의 B1, B2 → B2-0 → B3 → 시드 2
- 결과 파일을 읽는 법
- 흔한 문제와 해결: GPU 메모리 부족이면 `batch_size`를 줄인다. 세션이 끊기면 같은 셀을 다시 실행한다(`resume`). 특징을 다시 뽑아야 하면 캐시 폴더를 지운다.
- 대략의 시간: 실측이 없으면 "미측정"이라고 쓴다.

## 7. 필수 확인 (이것만 한다)

1. **작은 학습 시험**: 이 PC에서 CPU용 torch와 torchvision을 설치해, main20과 ctrl5 각각 train 4개, val 2개로 4개 프리셋을 모두 1에폭 돌린다. 그리고 `eval`, `swaptest`, `predict`, `score`(val 정답을 가짜 private로 사용), `summary`까지 끝까지 실행되는지 본다. 목적은 코랩에서 모양 오류나 경로 오류가 나지 않게 하는 것이다. 설치가 불가능하면 할 수 있는 부분만 돌리고, 돌리지 못한 부분을 보고서에 적는다.
2. **채점 시험 3개** (val 정답으로):
   - (a) 정답을 그대로 예측으로 넣으면 확장 F와 기본 F가 모두 1이고 DOA 오차가 0이다.
   - (b) 모든 방위각을 앞뒤로 뒤집어(`az → 180 − az`를 −180~180으로 정리) 넣으면, 기본 F는 그대로이고 확장 F는 떨어진다.
   - (c) 정답이 빈 클립에 가짜 예측을 하나 넣으면 오검출로 세어져 F가 떨어진다.
3. **코드 읽기 확인 2개** (고치기 전 원본 기준, 보고만 한다):
   - (a) 영상 전처리가 쓰는 ResNet 가중치와 변환을 확인한다. 가운데를 자르는지, 자른다면 원래 수평 시야 100° 중 몇 도가 남는지 계산한다. 원근 영상이라 각도는 픽셀에 비례하지 않으므로 `2·atan((자른 폭 / 줄인 폭) · tan 50°)`로 계산한다.
   - (b) 디코더 호출에 `memory_mask`나 위치 인코딩이 있는지 확인한다.

## 8. 산출물

```
baseline/seld_challenge/   고친 코드, run.py, score.py, configs/, colab_run.ipynb
README_baseline.md         코랩 사용 설명서
CHANGES.md                 원본 대비 바꾼 곳 (파일, 줄, 이유)과 정한 기본값
CHECK_REPORT.md            7절 확인 결과
internal/                  ctrl5, fix20 전송용 zip (비공개)
```

## 9. 하지 말 것

- 실제 학습 (7절 1번의 1에폭 시험 외)
- 7절 외의 검증, 테스트 프레임워크, 그림
- 원본 베이스라인과 데이터 파일 수정
- 카메라 자세를 모델 입력으로 넣기
- 공개 zip 수정, 내부 zip을 공개 폴더에 두기
- 사용자에게 질문하고 멈추기