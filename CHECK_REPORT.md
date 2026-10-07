<!-- [변경 이력 시작]
  2026-10-03  최초 생성: 필수 확인 결과(7.1 소규모 학습은 보류로 기록, 7.2 채점 시험 3개, 7.3 코드 읽기 확인 2개)
  2026-10-03  7.1을 conda 환경(CPU)에서 실제로 돌린 소규모 학습 시험 결과로 교체, 시험 중 찾아 고친 문제 3가지 기록
  2026-10-07  변경 이력 주석 추가
[변경 이력 끝] -->

# CHECK_REPORT: 필수 확인 결과 (MODEL_CODE_CH.md 7절)

## 7.1 작은 학습 시험: 했음 (conda 환경 `seld_test`, CPU)

처음에는 사용자 지시로 보류했다가, 이후 "conda 환경을 만들어서 소규모 실험을 수행"하라는 지시에 따라 실시했다.

**환경**: conda 환경 `seld_test` (python 3.11.17, torch 2.14.1+cpu, torchvision 0.29.1+cpu, librosa 0.11.0, opencv 5.0, pandas 3.0, scipy 1.17). 재현용 파일은 `build/conda_env_seld_test.yml`, `build/pip_freeze_seld_test.txt`. GPU 없이 CPU로만 돌렸다.

**미니 데이터** (`build/make_mini.py`): main20, ctrl5, fix20 각각 train 4개, val 2개, test 2개. test는 val 클립 2개를 `test_0000x` 이름으로 복사하고 그 정답을 가짜 private으로 썼다. ctrl5 val에는 정답이 빈 클립(`val_00000`)을 일부러 넣었다. 실제 `prepare`가 읽는 것과 같은 zip 구조로 만들었다.

**결과**
- (1) **전체 실행**: main20, ctrl5 × B1, B2, B2-0, B3, 에폭 1, 모든 단계 `prepare → features → train → eval → swaptest(B1 제외) → predict → score`, 마지막에 `summary`. 총 55단계가 **모두 exit 0**, 실패 없음 (`build/mini_all.ps1`).
- (2) **gaps 계산**: 접미사 없는 실행 이름이 되도록 기본값을 1에폭·배치 2로 둔 설정(`build/mini_base.yaml`)으로 train/eval/swaptest를 다시 돌려 `gaps.csv`를 만들었다. main20 B2−B1, B2−(B2−0), B3−B1, ctrl5 같은 3개, 그리고 `main20-ctrl5` 차이까지 계산됨 (`build/mini_gaps.ps1`).
- (3) **eval_dataset=fix20**: main20과 ctrl5(길이 50) 모델을 fix20 val(길이 200)로 평가해서 2클립 채점 성공.
- (4) **제외 목록**: main20은 `val_00088`이 미니 val에 없어 클립 수 그대로(2), ctrl5는 `parent_map`으로 `val_00001`로 바뀌어 1클립만 채점(이 클립이 정답 없는 클립뿐이라 F = NA).
- (5) **영상 교체**: `shuffle`에서 4개 클립이 모두 다른 클립의 영상을 받았다(원래 번호 [3,0,1,2]), `zeros`의 합은 0. B2−0(영상 0으로 학습한 모델)은 `real/shuffle/zeros` 결과가 같게 나와 의도대로다.
- (6) **정답이 빈 클립**(ctrl5 val_00000): 특징 추출, 학습 손실, 채점 모두 문제없이 처리.
- (7) **B3**(`video_crop=full` 특징 98차원, `av_time_window=2` 마스크): 학습, 평가, 예측 모두 오류 없이 동작.
- (8) **이어 하기**: 이미 한 번 학습한 main20 B2를 같은 설정으로 `train`을 다시 돌리니 `last.pth`에서 이어져(남은 에폭 0) 학습 없이 요약만 쓰고 끝남.

**이 시험에서 찾아서 고친 문제**
1. 제외 목록을 켜고 `eval`하면 같은 실행의 `scores_val_real.json`을 덮어써서 `gaps`가 틀어졌다 → 제외 평가는 `scores_val_real_excl.json`으로 따로 저장하고, `summary`/`gaps`는 제외 여부(`exclude_list` 열)별로 계산하도록 고침 (`run.py` `excl_tag`).
2. 학습 클립 수가 배치 크기보다 작으면 에폭당 갱신 횟수가 0이 되는 문제 → 알기 쉬운 에러를 내도록 함.
3. 출력의 `Onscreen=NA%` 표기 → `NA`로 고침 (`score.py` `one_line`).

**이 시험으로 알 수 없는 것**
- 점수는 의미가 없다(클립 4개, 1에폭이라 F가 0~5%). 코드가 끝까지 도는지만 본 것이다.
- **GPU에서의 동작**(CUDA 장치 이동, 메모리 부족 여부, 배치 64/256의 실제 메모리 사용량)은 확인하지 못했다. 코랩에서 첫 실행 때 확인해야 한다.
- 실제 크기 데이터에서의 시간과 메모리: **미측정**. 참고로 CPU에서 영상 특징 추출은 200프레임 클립당 약 15초였다(main20, `center`).
- Windows(CPU)에서 돌렸고 코랩은 Linux다. 경로 구분자 문제는 `os.path`만 써서 없을 것으로 보지만 확인은 못 했다.

## 7.2 채점 시험 3개 (val 정답, `check_scoring.py`)

| 시험 | main20 val (123개, 길이 200) | ctrl5 val (492개, 길이 50) |
|---|---|---|
| (a) 정답 = 예측 | ext F 100%, basic F 100%, DOA 오차 0.00, 상대거리 0.00, onscreen 100% → **통과** | 같음 → **통과** |
| (b) 앞뒤 뒤집기 (`az → 180−az`) | basic F 100% (그대로), ext F 10.2%, DOA 90.4° → **통과** | basic F 100%, ext F 17.9%, DOA 83.1° → **통과** |
| (c) 정답이 빈 클립에 가짜 예측 1개 | (main20 val에는 빈 클립이 없음) | 빈 클립 `val_00000`에 class 0 예측 1개: F 1.000000 → 0.999995, class 12면 0.996865 → **통과** (오검출로 세어짐) |

참고: 정답 없는 클래스(9, 11)를 평균에 넣으면 정답 그대로 예측해도 F_ext가 0.846이 된다. 그래서 `exclude_absent_classes`가 기본 true다.

## 7.3 코드 읽기 확인 (고치기 전 원본 기준)

### (a) 영상 전처리: 가운데를 자르고, 수평 시야는 100° 중 약 60°만 남는다

- `extract_features.py`(원본 54행)는 `ResNet50_Weights.DEFAULT`를 쓴다. torchvision 소스(0.29.1)에서 `DEFAULT = IMAGENET1K_V2`이고 변환은 `ImageClassification(crop_size=224, resize_size=232)`, 즉 **짧은 변을 232로 줄이고(양선형) 가운데 224×224를 자르는** 것이다.
- `utils.py`의 `load_video`(원본 124행)는 프레임을 먼저 **360×180**(너비×높이)으로 줄인다. 짧은 변(높이 180)을 232로 줄이면 너비는 360×232/180 = **464**가 되고, 가운데 **224**만 남는다. 남는 가로 비율은 224/464 = **0.483**이다.
- 원근 영상의 각도는 픽셀에 비례하지 않으므로 `2·atan(0.483 · tan 50°) = 2·atan(0.5754) =` **59.8°**다. 즉 원래 수평 시야 100° 중 **약 60°만 보이고, 좌우 각각 약 20°씩(합 40°)이 잘린다.** (세로는 232 중 224가 남아 거의 그대로, 약 65.8°/67.7°.)
- 영향: 소리 방향이 화면 안(`onscreen=1`)이어도 가장자리 약 40°에 있으면 영상 모델 눈에는 안 보인다. `video_crop=full`이 이 부분을 보완하는 옵션이다.
- 참고: 모양을 맞추려고 360×180(2:1)으로 줄이는 것은 640×360(16:9) 영상을 가로로 줄여 찌그러뜨린다. 원본도 같다. `full`은 448×224(2:1)이라 같은 방식으로 찌그러진다.

### (b) 디코더 호출에 memory_mask나 위치 인코딩이 있는가: 둘 다 없다

- 원본 `model.py` 125행: `fused_feat = self.transformer_decoder(audio_feat, vid_feat)`. `memory_mask`, `tgt_mask`, key padding mask 인자가 없다.
- 원본 `model.py` 전체에 위치 인코딩(positional encoding)이 없다(`memory_mask`, `pos` 등을 검색해 일치하는 것이 없음). 시간 순서는 GRU를 거친 오디오 쪽 표현에만 들어 있고, 영상 특징은 `Linear`만 거쳐서 디코더의 교차 주의에 들어간다.
- 결과: 영상 쪽 프레임 t가 오디오 쪽 프레임 t와 짝지어진다는 정보가 구조에 없다. 오디오 프레임 하나가 영상 **모든 프레임**(5초 또는 20초 전체)을 똑같이 볼 수 있다. 영상의 시간 위치는 모델이 데이터에서 알아내야 한다. `av_time_window=k`는 이 점을 마스크로 고치는 옵션이다.

## 7.4 이 확인에서 나온 그 밖의 발견

- 원본은 onscreen 예측을 `int(value[4])`로 저장해서 확률이 거의 항상 0이 된다(`CHANGES.md` 2절). 고쳤다.
- 원본 채점은 정답이 없는 파일, 마지막 정답 프레임 뒤의 예측을 세지 않는다(이전 작업의 `NOTES.md`). `clip_len`으로 고쳤다.
- 원본 `load_video`는 우리 영상에서 프레임 수를 3분의 1로 줄인다(200 → 67). 고쳤다.
