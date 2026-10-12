<!-- [변경 이력 시작]
  2026-10-13  최초 생성: NTU 시스템을 챌린지 데이터셋으로 학습하는 방법, 원본에서 바꾼 곳, 확인한 것과 못 한 것
  2026-10-13  코랩 학습 노트북(colab_ntu.ipynb) 사용법 추가
[변경 이력 끝] -->

# NTU 시스템을 챌린지 데이터셋으로 학습하는 버전 (`ntu_challenge`)

DCASE 2025 Task 3 참가팀 NTU(Yeow, Tan, Peksi, Gan)의 소리만 쓰는 시스템 [itsjunwei/NTU_SNTL_Task3](https://github.com/itsjunwei/NTU_SNTL_Task3) (커밋 `0e4f207`, MIT 라이선스)을
우리 챌린지 데이터셋(`main20`, `ctrl5`)으로 **학습만** 돌릴 수 있게 고친 것입니다. 원본 클론(`baseline/NTU_SNTL_Task3/`)은 고치지 않았고, 이 폴더에 필요한 파일을 복사해 고쳤습니다.

**하는 일**: zip 풀기(선택) → 특징 뽑기 → 학습 → 매 에폭 val 채점으로 최고 모델 저장.
**하지 않는 일**: test 예측과 제출 파일 만들기, 이어 하기(resume), 추가 생성 데이터(S1) 학습, 영상 사용(NTU 시스템은 소리만 씁니다), 새 기능 추가.

## 쓰는 법

**코랩**: `colab_ntu.ipynb`를 열고 위에서부터 셀을 실행합니다(설정 칸만 바꾸면 됩니다). 학습이 한 셀(`main.py`)이고, 마지막 셀이 최고 모델과 val 점수를 보여 줍니다. 이어 하기가 없어서 학습 중 세션이 끊기면 학습을 처음부터 다시 해야 합니다(특징 캐시는 드라이브에 남습니다).

```
pip install -r requirements.txt

python main.py --dataset main20 --data_root /content/data --out_root /content/drive/MyDrive/.../seld_runs \
               --zip_dir /content/drive/MyDrive/.../dataset/main20
python main.py --dataset ctrl5  --data_root /content/data --out_root ... --zip_dir .../internal
```

| 인자 | 뜻 |
|---|---|
| `--dataset` | `main20`(라벨 200프레임) 또는 `ctrl5`(50프레임) |
| `--data_root` | `{data_root}/{dataset}/{train,val}/{audio,labels}/` 가 있는(또는 zip을 풀) 곳. 코랩에서는 로컬 디스크 |
| `--out_root` | 결과(`runs/`)와 특징 캐시(`features/`)가 저장되는 곳 |
| `--zip_dir` | 주면 `train`, `val` zip을 `data_root`에 풀어 줌. main20은 `train.zip`, `val.zip`, ctrl5는 `internal_ctrl5_train.zip`, `internal_ctrl5_val.zip`. 이미 풀려 있으면 건너뜀 |
| `--seed`, `--nb_epochs`(100), `--batch_size`(32), `--accum_batch`(auto), `--learning_rate`(1e-3), `--weight_decay`(1e-4), `--val_freq`(2), `--select_metric`(F_ext) | 학습 설정. 기본값은 NTU 원본 값 |
| `--ms --gamma --iv --ipd --slite`, `--cutout --itfm --freqshift --filtaug --compfreq --alltrans` | NTU 원본의 특징, 증강 옵션. 기본은 모두 꺼짐(왼쪽, 오른쪽 로그멜 2채널) |
| `--local_tmp` | 에폭마다 쓰는 val 예측 csv 위치(로컬 디스크). 기본은 시스템 임시 폴더 |
| `--exp` | 실행 이름 꼬리표 |

결과: `{out_root}/runs/ntu_{dataset}_s{seed}[_exp]/` 에 `best_model.pth`, `train_log.csv`(에폭별 손실과 val 점수), `train_summary.json`, `config_used.json`.
특징 캐시: `{out_root}/features/{dataset}/ntu_{mel64_..._dnorm}/` (500클립씩 묶음 파일 + `stats_train.npz`). 이미 있으면 다시 만들지 않습니다.

학습 방식은 원본 그대로입니다: AdamW(가중치 감쇠 1e-4, bias/BN/LN 제외), OneCycleLR, 기울기 누적, 100에폭, 멀티 ACCDOA(트랙 3개), ADPIT 손실, 거리 정규화(dnorm).
모델 고르기와 점수는 `seld_challenge`와 같은 방식입니다: val을 `F_ext`(방위각을 접지 않음, 챌린지 점수)와 `F_basic`(접음, DCASE 공식)로 채점하고 `F_ext`가 가장 높은 에폭을 저장합니다.

## 원본에서 바꾼 곳

**필수 수정** (우리 데이터에서 돌아가거나 점수가 맞으려면 필요)

| # | 무엇 | 파일 | 이유 |
|---|---|---|---|
| 1 | **출력 x를 `clamp(0, 1)`로 자르던 것을 `clamp(-1, 1)`로** | `utils.py` `get_accdoa_labels`, `get_multiaccdoa_labels` (원본 659, 680, 695, 710줄) | DCASE 원본 데이터는 방위각이 앞쪽(−90~90°)으로 접혀 있어서 x=cos(방위각)≥0이다. 우리 라벨은 접지 않은 −180~180°라 뒤쪽 소리는 x<0이다. 자르면 뒤쪽 소리를 낼 수 없다 |
| 2 | 폴더와 이름 규칙: `stereo_dev/dev-*`, `metadata_dev/dev-*`, fold 이름 → `{dataset}/{split}/{audio,labels}/`와 split 이름 | `extract_features.py`, `data_generator.py`, `main.py` | 우리 데이터셋 구조 |
| 3 | 라벨 길이를 데이터셋에 따라 자동으로 (main20 200, ctrl5 50) | `main.py` `LABEL_LEN` | 원본은 50 고정(5초) |
| 4 | 채점을 우리 `score.py`/`metrics.py`로 (val 정답 폴더, 클립 길이 기준 프레임 수, 접지 않은 방위각 오차, 예측 없는 클립은 빈 예측으로) | `score.py`, `metrics.py`, `seld_label_utils.py` (`baseline/seld_challenge`에서 그대로 복사) | 원본 `metrics.py`는 fold 구조를 가정하고 방위각을 접어서 비교한다. 챌린지 점수와 같은 계산이어야 한다 |
| 5 | 모델 고르기를 test가 아니라 val 점수로 | `main.py` | 원본은 fold4(= 우리 test)로 골랐다. 우리 test 정답은 비공개 |
| 6 | 예측 파일의 `onscreen`을 `int(value[4])`가 아니라 `> 0.5`로 | `utils.py` `write_to_dcase_output_format` (원본 909줄) | 소리만 쓰는 이 시스템에서는 항상 0이라 결과는 같지만, 영상 모델 버그를 그대로 두지 않으려고 `seld_challenge`와 맞췄다 |
| 7 | 정규화 통계를 **train에서만** 구함. 원본은 fold3, 4, 5 전부(test fold 포함)로 `StandardScaler`를 맞췄다 | `extract_features.py` `fit_stats`, `data_generator.py` | val, test 정보가 학습에 새지 않게. 계산은 `StandardScaler`와 같다(왼쪽, 오른쪽 채널은 통계 공유) |
| 8 | 원본 코드가 그대로는 돌지 않는 곳: `params['finetune']`(정의된 적 없음, `main.py:272`, `data_generator.py:111`)와 `params['val_batch_size']`(인자는 `--val_batch`, `main.py:122, 188`)에서 KeyError | `main.py`, `data_generator.py`, `parameters.py` | `finetune` 옵션은 뺐고 `val_batch_size`는 `parameters.py`에 둠 |
| 9 | 우리 train 폴더에는 추가 생성 데이터(`train_s1_*`)가 같이 풀려 있을 수 있어 `train_NNNNN`만 사용 | `extract_features.py` `clip_names` | 이 버전은 기본 데이터(S0)만 학습 |

**정리한 것** (기능은 같음)

- 클립마다 `.pt` 파일 대신 500클립 묶음 파일로 캐시(드라이브에서 작은 파일 수천 개 읽기가 느림). 정규화는 불러올 때 적용.
- `wandb`, `tensorboard`, `torchinfo`, `rich`, `scikit-learn`, `joblib` 의존성 제거. 학습 기록은 `train_log.csv`.
- jackknife 신뢰구간 제거(원본도 기본 꺼짐).
- 가중치 갱신 단위(`accum_batch`)를 `auto`로: 12800 ÷ 라벨 프레임 수(main20 64, ctrl5 256). `seld_challenge`와 같은 규칙이고, 한 번 갱신에 보는 라벨 프레임 수가 두 데이터셋에서 같다.
- 윈도우에서는 `num_workers=0`.

**그대로 복사한 파일**: `model.py`, `loss.py`, `training_utils.py`(증강) — NTU 원본. `utils.py`는 위 1, 6번과 `setup()` 제거, 예측 파일 이름을 클립 이름으로 받게 한 것 외에는 원본 그대로.
**쓰지 않는 원본 파일**: `left_right_swap.py`(DCASE 원본 데이터를 만들 때 쓰는 좌우 바꾸기), `track_results.py`, `dist_norm_params.ipynb`(거리 정규화 상수 계산).

## 확인한 것 (작은 시험 데이터로, CPU)

- main20(200프레임)과 ctrl5(50프레임)가 끝까지 돈다. zip 풀기, 특징 뽑기, 학습, val 채점, 최고 모델 저장까지.
- `--ms --iv --gamma --itfm`(6채널 특징과 증강)도 돈다.
- 정답 라벨을 모델 출력 모양으로 바꿔서 예측 파일로 쓰고 정답과 채점하면 F=0.969, 각도 오차 0.06°. 뒤쪽 소리(|방위각|>90°)는 정답 73.2%, 복원 73.4%로 보존된다. 1번 수정이 없으면 뒤쪽 소리가 모두 사라진다.

## 확인하지 못한 것

- **전체 데이터로 학습한 적이 없다.** 학습 시간, 메모리, 점수는 모른다. main20 train 3,000클립은 특징이 약 2.5GB(float32, 2채널)이고 `--ms --iv --gamma`를 켜면 채널이 6개라 약 7.5GB다.
- 거리 정규화 상수(`d_mean` 등)는 원본 값 그대로다. 우리 train 라벨 거리(평균 1.95m, 표준편차 0.68m, 600클립 표본)와 비슷해서 그대로 뒀다.
- 원본의 100에폭, OneCycleLR 설정이 `main20`에 적절한지는 시험하지 않았다. 우리 `seld_challenge`는 200에폭, 고정 학습률을 썼으므로 점수를 직접 비교할 때는 이 차이를 감안해야 한다.
