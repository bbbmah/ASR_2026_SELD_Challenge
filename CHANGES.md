<!-- [변경 이력 시작]
  2026-10-03  최초 생성: 원본 베이스라인 대비 변경 목록(필수 수정 10개, 추가로 찾은 것, 새 모델 옵션, 정한 기본값)
  2026-10-03  소규모 시험 결과 반영: 제외 목록 평가를 별도 파일(_excl)로 저장, 학습 클립이 배치보다 적을 때 에러 항목 추가
  2026-10-07  6절 추가: 추가 학습 데이터(S1) 연동 변경(train_stages, updates, 필터, datainfo, data_gaps)
  2026-10-07  변경 이력 주석 추가
[변경 이력 끝] -->

# CHANGES: 원본 베이스라인 대비 바꾼 곳

원본은 `baseline/DCASE2025_seld_baseline/`(그대로 둠), 고친 코드는 `baseline/seld_challenge/`다. 줄 번호의 "원본"은 원본 파일, "새"는 고친 파일 기준이다. 데이터 파일은 고치지 않았다.

## 1. 필수 수정 (CLAUDE 지침 3절)

| # | 무엇 | 파일, 줄 | 이유 |
|---|---|---|---|
| 1 | 영상 프레임 간격: fps를 30으로 고정하고 3장마다 1장 쓰던 것을 실제 fps로 `round(fps/10)` 계산. 특징을 뽑은 뒤 영상 프레임 수가 라벨 프레임 수와 다르면 에러 | `utils.py` `load_video` (원본 115-116 → 새 66-98, 간격 계산 82). `expected_frames` 검사는 98 근처 | 우리 영상은 이미 10fps다. 원본 코드로는 200프레임 영상이 67프레임이 된다(확인함). |
| 2 | 방위각 오차를 원형 차이 `|((a-b+180) mod 360)-180|`로 계산. 접는 방식은 옵션으로 남김 | `seld_label_utils.py` `circular_az_difference`(110), `az_difference`(116), `least_distance_between_gt_pred(..., fold=False)`(123). 원본은 `utils.py` 467-479, 700 | 접지 않은 방위각을 접어서 비교하면 앞뒤를 틀려도 오차 0이 된다. |
| 3 | 폴더와 이름 규칙: fold 이름과 `stereo_dev/dev-*` 경로를 `{dataset}/{split}/{audio,video,labels}/`와 split 이름(`train`, `val`, `test`)으로 | `config.py`(`dataset_dirs`, `SPLITS`), `extract_features.py`, `data_generator.py`, `run.py`. 원본 `parameters.py` 72-73(`dev_train_folds` 등), `extract_features.py` 72/74/99/101/142/144, `data_generator.py` 96-101/125-130 | 우리 데이터 구조에 맞춤. test 라벨은 `{dataset}/private/test_labels/`. |
| 4 | 라벨 길이를 데이터셋에 따라 자동으로 (main20·fix20 200, ctrl5 50) | `config.py` `LABEL_LEN`, `load_config` | 원본 `parameters.py` 55 고정값 50 |
| 5 | 정답 폴더 읽기: 하위 폴더를 가정하던 것을 csv가 바로 든 폴더(`val/labels/`)도 읽게 | `metrics.py` `ComputeSELDResults.__init__`(새 224-229). 원본 219-225 | |
| 6 | 모델 고르기: 매 에폭 점수를 test가 아니라 **val**로 계산하고 그 점수로 최고 모델 선택 | `run.py` `stage_train`. 원본 `main.py` 109/121/138/165 | test 정답은 비공개 |
| 7 | 정답이 하나도 없는 클래스를 평균 F에서 뺄 수 있게(`exclude_absent_classes`, 기본 true) | `metrics.py` `SELDMetrics`(새 49, 89-90). 원본 87 | val에는 9번(악기), 11번(종) 정답이 없다 |
| 8 | 배치 크기를 설정으로. 기본 `auto` = 12800 ÷ 라벨 프레임 수 (main20 64, ctrl5 256) | `config.py` `FRAMES_PER_BATCH`, `load_config`. 원본 `parameters.py` 63 | 한 배치의 프레임 수와 에폭당 갱신 횟수를 두 데이터셋에서 맞춤 |
| 9 | 채점 프레임 수: 정답 파일의 마지막 프레임 대신 **클립 길이** | `metrics.py` `ComputeSELDResults`(새 214, 229, 269). 원본 223 `nb_ref_frames = max(frame)` | 원본은 정답이 없는 파일과 마지막 소리 이후의 예측을 오검출로 세지 않는다 (`CHECK_REPORT.md`) |
| 10 | `nb_workers`를 설정으로(기본 2). 기존 체크포인트는 쓰지 않음(처음부터 학습) | `configs/base.yaml` `num_workers`, `config.py`. 원본 `parameters.py` 64 | |

## 2. 지침에 없지만 읽다가 발견해서 고친 것

| 무엇 | 파일, 줄 | 이유 |
|---|---|---|
| **onscreen 예측이 거의 항상 0으로 저장되던 버그**: `int(value[4])`로 시그모이드 확률(0~1)의 소수점을 버렸다. 0.5 기준으로 바꿈 | `utils.py` `write_to_dcase_output_format`(새 474). 원본 614 | 안 고치면 onscreen 정확도가 "정답이 0인 비율"이 되어 의미가 없다 |
| 예측 csv가 없는 클립을 "아무것도 예측 안 함"으로 채점 | `metrics.py` `get_SELD_Results`(새 261-273) | 원본은 있는 예측 파일만 채점해서 점수가 부풀 수 있다 |
| jackknife 신뢰구간 제거(함수 `jackknife_estimation`은 `seld_label_utils.py`에 남김), `print_results`, `setup()`(텐서보드 폴더 만들기) 제거 | `metrics.py`, `utils.py` | 우리 흐름에서 쓰지 않음. 실행 폴더와 로그는 `run.py`가 관리 |
| 제외 목록을 쓴 평가는 `scores_*_excl.json`으로 따로 저장, `summary`/`gaps`는 `exclude_list`별로 계산 | `run.py` `excl_tag`, `stage_summary` | 제외 없는 결과를 덮어쓰지 않도록 (소규모 시험에서 발견) |
| 학습 클립 수 < 배치 크기이면 에러 | `run.py` `stage_train` | 갱신 횟수 0으로 인한 0 나눗셈 방지 |
| `main.py`, `evaluate.py`, `inference.py`는 `run.py`의 `train`, `predict`, `eval` 단계로 넘기는 얇은 껍데기로 바꿈 | 각 파일 | 설정 파일, 프리셋, 이어 하기, 두 방식 채점이 `run.py`에 있어 중복을 피함 |
| torch 없이 쓰는 라벨 읽기와 각도 함수를 `seld_label_utils.py`로 분리 | `seld_label_utils.py` | `score.py`/`metrics.py`가 torch 없이 돌게 함 (이 PC에서도 시험 가능) |

## 3. 새로 만든 모델 옵션 (지침 4절)

| 옵션 | 구현 위치 |
|---|---|
| `modality` audio / audio_visual | 원래 있던 것 |
| `video_input` real / zeros | `model.py` forward (새 133-134): 영상 특징을 `zeros_like`로 바꾼 뒤 **같은 구조**(Linear, 디코더)를 그대로 통과 |
| `video_crop` center / full | `extract_features.py` `_load_resnet`(80-94): center = 원래 전처리, full = 224×448로 줄여 자르지 않음 → 특징 7×14, 채널 평균 후 98차원. `config.py`가 `resnet_feature_size`(49/98)를 정해 `model.py`의 `Linear` 입력 크기에 쓴다 |
| `av_time_window` none / k | `model.py` `_memory_mask`(93) + 디코더 `memory_mask=`(138). bool 마스크, True = 못 봄. 오디오 프레임 수 == 영상 프레임 수 assert(135) |
| 프리셋 B1, B2, B2-0, B3 | `config.py` `PRESETS` |

카메라 자세는 어떤 옵션에서도 모델 입력에 넣지 않는다.

## 4. 새 파일

`run.py`(단계 실행), `config.py`와 `configs/base.yaml`(설정), `score.py`(채점), `seld_label_utils.py`, `colab_run.ipynb`, `check_scoring.py`(채점 시험), `configs/exclude_val_fix20.txt`. 프로젝트 루트: `README_baseline.md`, `CHECK_REPORT.md`, `internal/`(ctrl5, fix20 전송용 zip), `build/make_internal_zips.py`.

## 5. 정한 기본값 (애매해서 임의로 정한 것)

- **특징 캐시는 묶음 파일**: 클립마다 .pt 대신 500개씩 묶어 `{out_root}/features/{dataset}/{audio|video_center|video_full|labels_adpit}/{split}_{000}.pt`로 저장한다. 드라이브에서 작은 파일 수천 개를 읽으면 느리기 때문이다. 학습할 때 전부 메모리에 올린다(main20 train 오디오 특징 약 1.5GB).
- **결과 위치**: `{drive_root}/seld_runs/{runs,features,results}`. 기본 `drive_root`는 코랩 노트북에서 지정한다.
- **모델 고르는 기준**: `F_ext`(방위각 접지 않음). `F_basic`으로 바꿀 수 있다. 같은 값이면 나중 에폭을 쓴다(원본의 `>=`와 같음).
- **학습 중 val 채점에는 `exclude_list`를 쓰지 않는다**(전체 val). 비교할 때만 `eval`에서 뺀다.
- **오디오 특징**은 프리셋과 무관하게 한 번만 뽑는다. **영상 특징**은 `video_crop`별로 따로 저장한다.
- **훈련 데이터셋은 main20과 ctrl5만**. fix20은 train이 없으므로 `eval_dataset=fix20`으로 평가에만 쓴다.
- **swaptest의 shuffle**: 무작위 순서를 시드로 만들고 순서상 다음 클립의 영상을 받게 해서 어떤 클립도 자기 영상을 받지 않는다. 시드는 `1000 + seed`.
- **실행 이름**: `{dataset}_{preset}_s{seed}`. 프리셋과 다르게 덮어쓴 옵션, 기본값과 다른 `epochs/lr/batch_size/...`는 뒤에 `_win3-lr0.0005`처럼 붙는다.
- 제출 zip(`predict`)에는 예측 csv만 들어간다(클립당 하나, 이름은 `test_00000.csv` 형태).
- `exclude_list`는 클립 이름을 main20 기준으로 쓰고, ctrl5에서는 `ctrl5/private/parent_map.csv`로 조각 이름으로 바꾼다.


---

## 6. 추가 학습 데이터(S1) 연동 (학습 코드 변경)

원본 대비가 아니라 **직전 버전(S0 전용 학습 코드) 대비** 바뀐 곳입니다. 기본 설정(`train_stages=0`)에서는 결과가 이전과 같습니다(같은 시드로 다시 학습해 F_ext가 소수점 끝까지 일치함을 확인).

| 무엇 | 파일 | 이유 |
|---|---|---|
| 학습 데이터 단계 선택 `train_stages`(0, 0,1, 1), S1 zip 풀기 | `run.py`(`ensure_stage`, `stage_prepare`), `configs/base.yaml` | `generated/s1/`의 묶음 zip 25개를 `data_root`의 train 폴더에 푼다 |
| 단계별 특징 추출과 저장 접두사(`s1train_000.pt`), 클립 이름 필터 | `extract_features.py`(`clip_names`, `stage_key`) | S1 클립이 S0와 같은 폴더에 있어도 단계별로 따로 뽑고, 'train_*.pt' 패턴에 S1이 걸리지 않게 한다 |
| 영상 디코딩 스레드 미리 읽기 | `extract_features.py` | S1 클립 약 5,900개의 추출 시간을 줄임 |
| 묶음을 합치지 않고 목록으로 들고 인덱스로 읽기(`PackedTensors`), `keep` 필터, `audio_dtype=float16` | `data_generator.py` | S0+S1 특징 약 7GB를 하나로 합치면 잠깐 두 배의 메모리가 필요해 램이 모자랄 수 있다 |
| S1 필터: 가족 `train_family`, 거울 `train_mirror`, 비율 `train_fraction` | `run.py`(`stage_selection`) | 요인 분해와 학습 곡선 실험. 비율은 창 단위, 시드 고정, main20과 ctrl5가 같은 창 |
| 학습 길이를 총 갱신 횟수로: `updates` | `run.py`(`stage_train`), `config.py` | 데이터 양이 다른 실행을 같은 학습량으로 비교하기 위해(기존 약 9,200번) |
| 실행 이름에 데이터 꼬리표 `_S01-up9200` 등 | `config.py`(`data_tag`, `std_name`) | 기존 S0 실행과 덮어쓰거나 섞이지 않게. S0 실행 이름은 그대로 |
| 새 단계 `datainfo`(구성, 에폭, 갱신, 메모리 예상) | `run.py`(`stage_datainfo`) | 학습 전에 확인. 매니페스트만 읽어서 torch 불필요 |
| `summary`: 같은 데이터 구성끼리 `gaps.csv`, **S0 대비 효과표 `data_gaps.csv`** | `run.py`(`stage_summary`, `flatten`) | S0 대 S0+S1 비교. 점수 파일에 `data_tag`, `train_stages`, `updates` 등 기록 |
| 영상 교체 시험이 `PackedTensors`에서 동작 | `run.py`(`apply_eval_video`) | 데이터 구조 변경에 따른 수정 |
| 노트북: 설정 칸 6개, `datainfo` 셀, 추가 데이터 안내, `data_gaps` 출력, 추천 실험 순서 | `colab_run.ipynb`(만드는 코드 `build/make_train_notebook.py`) | |

정한 기본값: 노트북의 `train_stages` 기본값은 `0,1`, `updates`는 `9200`(S0와 같은 학습량으로 비교하려는 다음 실험에 맞춤). 예전처럼 쓰려면 `0`, `0`(비움)으로. 이전 노트북은 `build/colab_run_before_S1.ipynb`에 있다.
