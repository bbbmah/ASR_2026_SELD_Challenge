<!-- [변경 이력 시작]
  2026-10-03  최초 생성: 코랩 사용 설명서(순서, 설정의 뜻, 프리셋, 결과 읽는 법, 문제 해결)
  2026-10-03  소규모 시험 결과 반영: 주의 문구 수정
  2026-10-07  9절 추가(추가 학습 데이터 S1로 학습하기), 학습 시간 실측값 반영, 주의 문구와 준비물 폴더(generated/) 갱신
  2026-10-07  변경 이력 주석 추가
[변경 이력 끝] -->

# 베이스라인 사용 설명서 (코랩)

이 문서는 구글 코랩에서 혼자 실험을 돌리는 사람을 위한 설명서입니다. 코드는 `baseline/seld_challenge/`에 있고, **노트북 `colab_run.ipynb`의 설정 칸만 바꾸면** 모든 실험을 돌릴 수 있습니다. 원본 대비 바뀐 곳은 `CHANGES.md`, 확인 결과는 `CHECK_REPORT.md`를 보세요.

> 참고: 기존 데이터(S0)로는 코랩 GPU에서 실제 크기로 학습을 돌려 결과를 얻었습니다(`RESULTS_REPORT.md`). **추가 학습 데이터(S1)로 학습하는 부분(9절)은 PC(CPU)의 작은 데이터와 실제 S1 zip 몇 개로만 확인했고, 코랩에서 실제 크기로는 아직 돌려 보지 않았습니다.** 처음에는 9절의 "처음 한 번 시험" 순서대로 짧게 확인하세요.

## 1. 준비물 (드라이브 폴더)

프로젝트 폴더(`for_dataset`) 안에 아래가 있어야 합니다.

```
for_dataset/
  dataset/main20/ train.zip  val.zip  test_inputs.zip  private_test.zip   (공개 zip, main20)
  internal/       internal_ctrl5_{train,val,test,private}.zip
                  internal_fix20_{val,test,private}.zip                      (내부용, ctrl5와 fix20)
  dataset/ctrl5/private/parent_map.csv                                       (ctrl5 조각의 부모 대응표)
  generated/s1/   s1_main20_train_p000.zip ... , s1_main20_manifest.csv ...    (추가 학습 데이터 S1, colab_generate.ipynb 가 만든 것. 9절)
  baseline/seld_challenge/  (코드와 colab_run.ipynb)
```
결과, 체크포인트, 특징 캐시는 `for_dataset/seld_runs/` 아래에 자동으로 만들어집니다.

## 2. 코랩에서 처음부터 끝까지 돌리는 순서

1. 드라이브의 `baseline/seld_challenge/colab_run.ipynb`를 코랩으로 엽니다. 런타임 유형을 **GPU**로 바꿉니다(Pro+ 이면 더 좋은 GPU).
2. 셀을 **위에서부터 차례로** 실행합니다.
   1. 드라이브 연결과 패키지 설치. `DRIVE_ROOT`가 내 드라이브의 프로젝트 폴더 경로와 같은지 확인합니다.
   2. **설정 칸**: `dataset`, `preset`, `seed`, `epochs`, `batch_size`, `lr`, `eval_video` 등을 고릅니다.
   3. `prepare`: zip을 코랩 로컬 디스크에 풉니다.
   4. `features`: 오디오와 영상 특징을 뽑아 드라이브에 저장합니다.
   5. `train`: 학습합니다.
   6. `eval`과 `swaptest`: val 점수를 냅니다.
   7. `predict`: test 예측을 만들고 제출용 zip으로 묶습니다.
   8. `summary`: 모든 실행 결과를 한 표로 봅니다.
3. 다른 실험을 하려면 **2번 설정 칸만 바꾸고** 3번부터 다시 실행합니다. 이미 한 일(zip 풀기, 특징 뽑기)은 건너뜁니다.

명령줄로도 같은 일을 할 수 있습니다: `python run.py train --dataset main20 --preset B2 --set drive_root=... --set seed=2`.

## 3. 설정의 뜻

| 이름 | 뜻 | 가능한 값 |
|---|---|---|
| `dataset` | 학습할 데이터셋 | `main20`(카메라가 도는 20초), `ctrl5`(카메라 고정 5초) |
| `preset` | 모델 종류(아래 표) | `B1`, `B2`, `B2-0`, `B3` |
| `seed` | 난수 시드. 결과를 여러 번 비교하려면 1, 2, 3으로 | 정수 |
| `epochs` | 학습 횟수 | 기본 200 |
| `batch_size` | 한 번에 학습하는 클립 수 | `auto`(main20 64, ctrl5 256) 또는 숫자 |
| `lr` | 학습률 | 기본 0.001 |
| `eval_video` | 평가 때 영상을 바꾸는 방법 | `real`(그대로), `zeros`(0), `shuffle`(다른 클립 영상) |
| `eval_dataset` | 다른 데이터셋으로 평가 | 비움(같은 데이터셋), `main20`, `ctrl5`, `fix20` |
| `exclude_list` | 채점에서 뺄 클립 목록 | 비움 또는 `configs/exclude_val_fix20.txt` |
| `select_metric` | 가장 좋은 모델을 고르는 기준(`--set`으로) | `F_ext`(기본), `F_basic` |

세부 옵션은 `configs/base.yaml`에 한국어 주석과 함께 모두 있습니다. 개별 옵션(`modality`, `video_input`, `video_crop`, `av_time_window`)은 비워 두면 프리셋 값을 씁니다.

### 프리셋

| 프리셋 | 모델 | 영상 입력 | 쓰는 이유 |
|---|---|---|---|
| `B1` | 소리만 | 없음 | 기준선 |
| `B2` | 영상 + 소리 (원래 모델) | 진짜 영상, 화면 가운데만 | 영상을 쓰는 기준선 |
| `B2-0` | B2와 **같은 구조**, 영상 특징을 0으로 | 0 | 구조 효과와 영상 효과를 나누기 위한 대조군 |
| `B3` | B2 + 화면 전체 사용 + 시간 창 제한(±2프레임) | 진짜 영상, 화면 전체 | 영상을 더 잘 쓰게 한 모델 |

카메라 자세(어느 쪽을 보고 있는지)는 어떤 모델에도 입력으로 주지 않습니다.

## 4. 권장 실험 순서

1. main20: B1, B2
2. ctrl5: B1, B2
3. B2-0 (main20, ctrl5)
4. B3 (main20, ctrl5)
5. 시드 2 (위를 한 번 더). 가능하면 시드 3까지.

**처음 한 번 시험**: 설정 칸에서 `dataset=ctrl5`, `epochs=1`로 B1 → B2 → B3를 `train → eval → swaptest → predict`까지 돌려 보세요. 끝까지 도는지 확인한 뒤 진짜 실험을 하세요. (`epochs`를 바꾸면 실행 이름 끝에 `_ep1`이 붙어서 진짜 실험 결과와 섞이지 않습니다.)

## 5. 결과 파일 읽는 법

모든 결과는 `for_dataset/seld_runs/` 아래에 있습니다.

| 파일 | 내용 |
|---|---|
| `runs/{실행이름}/train_log.csv` | 에폭별 학습 손실, val 손실, val 점수 |
| `runs/{실행이름}/best_model.pth` | val 점수가 가장 좋은 모델 |
| `runs/{실행이름}/last.pth` | 가장 최근 에폭(이어 하기용) |
| `runs/{실행이름}/config_used.yaml` | 그 실행에 쓴 설정 전체 |
| `runs/{실행이름}/scores_val_{real,shuffle,zeros}.json` | val 점수(전체와 클래스별) |
| `runs/{실행이름}/submission_*.zip` | test 예측 csv 묶음 |
| `results/summary.csv` | 모든 실행의 점수 표 |
| `results/gaps.csv` | "영상 덕분에 오른 점수" |

실행 이름은 `{dataset}_{preset}_s{seed}`입니다(예: `main20_B2_s1`). 프리셋과 다르게 바꾼 옵션이 있으면 뒤에 붙습니다.

### 점수 두 종류

- **ext (확장 점수)**: 방위각을 접지 않고 비교합니다. 앞뒤를 틀리면 점수가 깎입니다. **순위와 모델 고르기에 이 점수를 씁니다.**
- **basic (기본 점수)**: 정답과 예측을 둘 다 앞쪽으로 접어서 비교합니다(DCASE 방식). 앞뒤를 틀려도 깎이지 않습니다.
- 열 이름: `F`(위치까지 맞힌 탐지 점수, 높을수록 좋음), `DOA`(방향 오차, 도, 낮을수록 좋음), `relDist`(상대 거리 오차), `onscreen`(화면 안/밖 정확도, 영상 모델만). 소리만 쓰는 B1은 onscreen이 `NA`(비어 있음)입니다.

### gaps.csv: 영상 덕분에 오른 점수

- `B2-B1` = (영상+소리) − (소리만). `B2-(B2-0)` = 진짜 영상 − 영상을 0으로 한 같은 구조. `B3-B1`.
- 계산에는 `onscreen`이 들어가지 않는 `F`만 씁니다. `F_ext_gap`과 `F_basic_gap` 두 가지가 나옵니다.
- 시드가 여럿이면 `seed=mean/std` 행에 평균과 표준편차가 나옵니다. `main20-ctrl5` 행은 main20의 격차에서 ctrl5의 격차를 뺀 값입니다(챌린지의 핵심 숫자).
- 모델이 영상을 실제로 쓰는지는 `swaptest`로 봅니다: `real`보다 `shuffle`과 `zeros`에서 점수가 크게 낮으면 영상을 쓰고 있다는 뜻입니다.

### 비교할 때 빼야 하는 클립

fix20에서 빠진 `val_00088`(피크 초과) 때문에, 세 데이터셋을 비교할 때는 이 창을 모두 뺍니다: `exclude_list=configs/exclude_val_fix20.txt`. ctrl5에서는 대응 조각 4개가 자동으로 빠집니다.

### fix20으로 평가하기

fix20은 학습 데이터가 없습니다. ctrl5나 main20으로 학습한 모델을 fix20 val로 평가하려면 `eval_dataset=fix20`을 고르고 `prepare → features → eval`을 실행하세요. 카메라 효과와 클립 길이 효과를 나누어 볼 때 씁니다.

## 6. 흔한 문제와 해결

| 증상 | 해결 |
|---|---|
| GPU 메모리 부족 (`CUDA out of memory`) | `batch_size`를 숫자로 줄입니다(예: 32, 16). 실행 이름에 `_bs32`가 붙어 다른 실행으로 취급됩니다. |
| 세션이 끊김 | 같은 셀(`train`)을 다시 실행합니다. `last.pth`에서 끊긴 에폭부터 이어 합니다. 코랩 로컬 데이터가 사라졌으면 `prepare`를 먼저 다시 실행합니다. |
| 특징을 다시 뽑아야 함(전처리를 바꿨거나 파일이 깨짐) | 드라이브의 `seld_runs/features/{dataset}/` 안에서 해당 종류 폴더(`audio`, `video_center`, `video_full`, `labels_adpit`)를 지우고 `features`를 다시 실행합니다. |
| `zip 을 찾을 수 없습니다` | `DRIVE_ROOT` 경로와 `dataset/main20/`, `internal/`에 zip이 있는지 확인합니다. |
| `영상 프레임 수 ... != 라벨 프레임 수` | 영상 파일이 깨졌거나 다른 데이터셋의 파일입니다. `prepare`를 지우고(`/content/data`) 다시 풉니다. |
| `swaptest` 에러 | 소리만 쓰는 B1에서는 정상입니다. B2, B2-0, B3에서만 실행하세요. |
| 학습이 너무 느림 | `num_workers`를 `--set`으로 늘립니다(기본 2). |

## 7. 시간

- 영상 특징 추출: CPU(내 PC)에서 main20 클립(200프레임) 하나에 약 15초였습니다. 코랩 GPU에서는 **미측정**입니다(S1 약 5,900클립의 추출 시간도 미측정).
- 학습(기존 S0, 코랩 GPU로 실측): 에폭당 약 11~12초, 한 실행 약 36~41분(main20과 ctrl5 모두, 200에폭). **S0+S1은 미측정**: 클립이 약 3배라 에폭당 시간도 약 3배(약 35초)로 예상되고, `updates=9200`이면 67에폭이라 총 시간은 S0와 비슷할 것으로 보입니다(추정).

## 8. 주최자용

- `score`: `private` 정답으로 test 예측을 채점합니다. `python run.py score --dataset main20 --preset B2 --set drive_root=...` (`internal_*_private.zip`/`private_test.zip`을 자동으로 풉니다).
- 임의의 폴더 채점: `python score.py --pred 예측폴더 --ref 정답폴더 --clip_len 200 --modality audio_visual --out scores.json`.
- test 정답과 `private_*.zip`은 **공개하면 안 됩니다.**

## 9. 추가 학습 데이터(S1)로 학습하기

`colab_generate.ipynb`가 만든 `generated/s1/`의 데이터(균일 격자 시간 창, 카메라 가족 F1~F5, 좌우 거울)를 기존 데이터(S0)에 더해 학습합니다. main20에서 먼저 쓰고, ctrl5도 같은 방법으로 됩니다(S1의 ctrl5 조각은 main20 창 하나당 4개).

### 설정 (노트북 2번 셀 또는 `--set`)

| 이름 | 기본값 | 뜻 |
|---|---|---|
| `train_stages` | `0` (노트북 기본 `0,1`) | `0` = 기존만(예전과 완전히 같음), `0,1` = S0 + S1, `1` = S1만 |
| `updates` | 비움 (노트북 기본 9200) | **총 갱신 횟수**로 학습 길이를 정함. 에폭 수 = 올림(updates / 에폭당 갱신 횟수). 비우면 `epochs` 사용 |
| `train_family` | 비움 | S1 중 쓸 카메라 가족. 예: `F1`(챌린지 정의), `F1,F4` |
| `train_mirror` | 비움 | S1 중 `1` = 거울 클립만, `0` = 거울 아닌 것만 |
| `train_fraction` | 1.0 | S1 창의 무작위 일부(시드 고정)만 사용. main20과 ctrl5가 같은 창을 고름 |
| `audio_dtype` | `float32` | 램이 모자라면 `float16`(오디오 특징 메모리 절반) |
| `generated_dir` | `{drive_root}/generated` | S1 zip이 있는 곳 |

`train_family`, `train_mirror`, `train_fraction`은 S1(단계 1 이상)에만 적용되고, S0만 쓸 때는 무시됩니다.

### 왜 "갱신 횟수"로 맞추나

기존 main20 학습은 에폭당 46회 × 200에폭 = 약 9,200번 갱신이었습니다. S0+S1(8,911클립)은 에폭당 139회이므로 같은 200에폭이면 갱신이 약 3배가 되어 "데이터가 늘어서"인지 "더 오래 학습해서"인지 가를 수 없습니다. `updates = 9200`이면 67에폭(약 9,300번)이라 학습량을 맞춰서 데이터 효과만 볼 수 있습니다. 갱신 횟수를 2배로 늘린 결과는 `updates = 18400`으로 따로 봅니다.

### 실행 이름과 결과

- 실행 이름 뒤에 데이터 구성이 붙습니다: `main20_B2_s1_S01-up9200`, `..._S01-famF1-up9200`, `..._S01-mir0-fr0.5`. **기존 S0 실행(`main20_B2_s1` 등)과 섞이거나 덮어쓰지 않습니다.**
- `summary`가 `results/data_gaps.csv`를 만듭니다: 같은 (데이터셋, 프리셋, 시드)에서 **기존 S0 실행 대비** val 점수 차이(`F_ext_delta`, `F_basic_delta`, `DOA_ext_delta` 등). 비교하려면 같은 프리셋과 시드의 기존 실행이 있어야 합니다.
- `gaps.csv`(영상 덕분에 오른 점수)는 이제 **같은 데이터 구성끼리** 계산하고 `data_tag` 열이 붙습니다.

### 새 단계: `datainfo` (학습 전에 꼭 확인)

`python run.py datainfo ...`(노트북 4-1 셀)은 이번 설정으로 학습에 실제로 들어갈 클립 수, 단계별 개수, 가족과 거울 비율, 에폭 수와 갱신 횟수, **메모리 예상**을 보여 줍니다(매니페스트만 읽어서 몇 초).

예: main20, S0+S1, `updates=9200` → 단계 0 2,999개, 단계 1 5,912개(F1 2,962 / F2 619 / F3 584 / F4 873 / F5 874, 거울 51%), 합계 8,911개, 에폭당 139회, 67에폭, 특징 메모리 약 7.2GB, 풀어 둔 클립 디스크 약 31GB.

### 처음 한 번 시험 (S1 학습)

1. `train_stages=0,1`, `updates=300` 정도로 B2 한 번만: `prepare → features → datainfo → train → eval`. 끝까지 도는지, `datainfo`의 메모리 예상이 런타임 램 안인지 확인합니다. (`features`는 S1 약 5,900클립이라 오래 걸리는 단계이고, 한 번 하면 드라이브에 캐시됩니다.)
2. 메모리가 모자라면 `audio_dtype=float16` 또는 고용량 램 런타임.
3. 문제가 없으면 `updates=9200`으로 진짜 실험을 합니다. 노트북 맨 아래에 추천 실험 순서(B1, B2, B2-0, B3 → 갱신 2배 → 요인 분해 → 시드)가 있습니다.

### 구현상 알아 둘 것

- S1 클립은 S0와 같은 `train/` 폴더에 풀리지만 이름이 다릅니다(`train_s1_NNNNN`). S1 특징 묶음은 `s1train_000.pt`처럼 **다른 접두사**로 저장되어 S0만 학습할 때 섞이지 않습니다.
- 특징 묶음을 하나의 큰 텐서로 합치지 않고 목록으로 들고 인덱스로 읽어서, 합칠 때 생기는 메모리 두 배 현상을 피했습니다.
- 데이터 로딩 방식이 바뀌어도 **S0만 쓰는 기본 설정의 결과는 이전과 완전히 같습니다**(같은 시드로 다시 학습해 F_ext가 소수점 끝까지 일치함을 확인).
- 영상 디코딩을 여러 스레드로 미리 읽어 GPU 계산과 겹칩니다.
