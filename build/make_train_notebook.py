# [변경 이력 시작]
#   2026-10-07  최초 생성: 학습 노트북 colab_run.ipynb를 만드는 스크립트(추가 학습 데이터 S1 설정 칸, datainfo 셀, data_gaps 출력, 추천 실험 순서)
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""baseline/seld_challenge/colab_run.ipynb (학습 노트북)를 만든다. python build/make_train_notebook.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}
C = []
C.append(md("""
# SELD 챌린지 베이스라인 실행 노트북

위에서부터 **셀을 순서대로** 실행하고, 2번 셀의 **설정 칸**만 바꾸면 됩니다. 자세한 설명은 `README_baseline.md`를 보세요.
런타임은 **GPU**(런타임 → 런타임 유형 변경)로 설정해 주세요. 세션이 끊기면 같은 셀을 다시 실행하면 이어서 합니다(`resume`).

**추가 학습 데이터(S1)를 쓰는 학습이 가능합니다.** `colab_generate.ipynb`가 만든 `generated/s1/`의 데이터를 기존 데이터(S0)에 더해 학습합니다. 설정 칸의 `train_stages`로 고릅니다(`0` = 기존만, `0,1` = 기존 + S1).
- **S0와 S0+S1을 공정하게 비교하려면 학습 길이를 "갱신 횟수"로 맞추세요**: 기존 main20 학습은 에폭당 46회 × 200에폭 = 약 9,200번 갱신이었습니다. `updates = 9200`을 쓰면 데이터가 늘어도 같은 갱신 횟수로 학습합니다.
- 기존 S0 결과(`main20_B2_s1` 등)는 그대로 있고, 새 실행은 이름 뒤에 `_S01-up9200`처럼 데이터 구성이 붙어 서로 덮어쓰지 않습니다.
"""))
C.append(md("""
## 1. 드라이브 연결, 폴더 위치, 패키지 설치
드라이브를 연결하고 코드 폴더로 이동합니다. 1분 이내.
`DRIVE_ROOT`는 `dataset/`, `internal/`, `generated/`, `baseline/`이 들어 있는 프로젝트 폴더 경로입니다. 다르면 고치세요.
"""))
C.append(code("""
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = "/content/drive/MyDrive/Colab Notebooks/ASR_2026-2/for_dataset"   #@param {type:"string"}
CODE_DIR = f"{DRIVE_ROOT}/baseline/seld_challenge"

import os
os.chdir(CODE_DIR)
!pip install -q librosa soundfile pyyaml tqdm
!nvidia-smi -L
!free -g | head -2
!df -h /content | tail -1
"""))
C.append(md("""
## 2. 설정 칸 (여기만 바꾸세요)
- `dataset`: 학습할 데이터셋 (`main20` = 카메라가 도는 20초, `ctrl5` = 카메라 고정 5초)
- `preset`: B1 = 소리만 / B2 = 영상+소리 / B2-0 = B2에서 영상을 0으로 / B3 = B2 + 화면 전체 + 시간 창
- `batch_size`: `auto`면 main20 64, ctrl5 256. GPU 메모리가 모자라면 숫자로 줄이세요.
- `eval_video`: 평가 때 영상을 `real`(그대로) / `zeros`(0) / `shuffle`(다른 클립 영상)로 바꿉니다.
- `eval_dataset`: 비우면 학습한 데이터셋으로 평가합니다. 다른 데이터셋으로 평가하려면 고르세요(예: fix20).

**추가 학습 데이터**
- `train_stages`: `0` = 기존 데이터만(예전과 같음), `0,1` = 기존 + S1, `1` = S1만.
- `updates`: **총 갱신 횟수**로 학습 길이를 정합니다(0이면 사용 안 함 = `epochs` 사용). S0와 비교할 때는 `9200`.
- `train_family`: S1 중 쓸 카메라 가족. 비우면 전부. `F1` = 챌린지 정의(sweep)만, `F1,F4` 처럼 여럿도 가능.
- `train_mirror`: S1 중 `1` = 좌우 거울 클립만, `0` = 거울이 아닌 것만, 비우면 둘 다.
- `train_fraction`: S1 창의 무작위 일부만 씁니다(0~1, 학습 곡선 실험용).
- `audio_dtype`: 램이 모자라면 `float16`(오디오 특징 메모리가 절반).
"""))
C.append(code("""
dataset = "main20"       #@param ["main20", "ctrl5"]
preset = "B2"            #@param ["B1", "B2", "B2-0", "B3"]
seed = 1                 #@param {type:"integer"}
epochs = 200             #@param {type:"integer"}
batch_size = "auto"      #@param {type:"string"}
lr = 0.001               #@param {type:"number"}
eval_video = "real"      #@param ["real", "zeros", "shuffle"]
eval_dataset = ""        #@param ["", "main20", "ctrl5", "fix20"]
exclude_list = ""        #@param ["", "configs/exclude_val_fix20.txt"]

# ---- 추가 학습 데이터 (기본값 = S0 + S1, 기존과 같은 갱신 횟수) ----
train_stages = "0,1"     #@param ["0", "0,1", "1"]
updates = 9200           #@param {type:"integer"}
train_family = ""        #@param ["", "F1", "F1,F4", "F2,F3,F4,F5"]
train_mirror = ""        #@param ["", "0", "1"]
train_fraction = 1.0     #@param {type:"number"}
audio_dtype = "float32"  #@param ["float32", "float16"]

import shlex
_sets = {"drive_root": DRIVE_ROOT, "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr, "eval_video": eval_video,
         "train_stages": train_stages, "train_fraction": train_fraction, "audio_dtype": audio_dtype}
if eval_dataset: _sets["eval_dataset"] = eval_dataset
if exclude_list: _sets["exclude_list"] = exclude_list
if updates: _sets["updates"] = updates
if train_family: _sets["train_family"] = train_family
if train_mirror != "": _sets["train_mirror"] = train_mirror
ARGS = shlex.join(["--dataset", dataset, "--preset", preset] + [x for k, v in _sets.items() for x in ("--set", f"{k}={v}")])
print(ARGS)
"""))
C.append(md("""
## 3. prepare: 데이터 풀기
드라이브의 zip을 코랩 로컬 디스크(`/content/data`)에 풉니다. 이미 풀려 있으면 건너뜁니다. 처음에는 **수 분**(main20 약 12GB) 걸릴 수 있습니다. 세션이 새로 시작되면 다시 풀어야 합니다.
`train_stages`에 1이 들어 있으면 **S1 zip(main20 약 17GB, 25개)도 풉니다**: 추가로 **10~20분**(미측정)과 로컬 디스크 약 21GB가 듭니다. 디스크가 모자라면(`df -h`) 런타임 디스크가 큰 유형을 쓰세요.
"""))
C.append(code("!python run.py prepare {ARGS}"))
C.append(md("""
## 4. features: 특징 뽑기
오디오와 영상 특징을 뽑아 **드라이브에 저장**(캐시)합니다. 이미 있으면 건너뛰므로, 처음 한 번만 오래 걸립니다(영상 특징은 GPU로도 **수십 분**, 미측정).
오디오 특징은 프리셋과 무관하게 한 번만, 영상 특징은 `video_crop`(center/full)마다 따로 저장됩니다.
**S1은 클립이 약 5,900개(S0의 약 2배)라서 이 셀이 가장 오래 걸립니다.** 영상 디코딩은 여러 스레드로 미리 읽어 GPU 계산과 겹칩니다. S1 특징은 `s1train_000.pt`처럼 다른 이름으로 저장되어 S0 특징과 섞이지 않습니다.
세션이 끊기면 같은 셀을 다시 실행하세요(끝난 묶음 파일은 건너뜁니다).
"""))
C.append(code("!python run.py features {ARGS}"))
C.append(md("""
## 4-1. datainfo: 학습 데이터 구성 확인 (학습 전에 꼭 보세요)
이번 설정으로 **실제로 학습에 들어갈 클립 수**, 단계별 개수, 카메라 가족과 거울 비율, **에폭 수와 갱신 횟수**, **메모리 예상**을 보여 줍니다. 몇 초.
- 메모리 예상이 런타임 램을 넘으면 `audio_dtype = "float16"`으로 바꾸거나, 고용량 램 런타임을 쓰세요(S0+S1 main20은 약 7GB).
- S0와 S0+S1을 비교한다면 두 쪽의 "약 N번 갱신"이 비슷한지 확인하세요(S0는 약 9,200번).
"""))
C.append(code("""
!python run.py datainfo {ARGS}
!free -g | head -2
"""))
C.append(md("""
## 5. train: 학습
학습합니다. 매 에폭 `last.pth`(이어 하기용)와 `best_model.pth`(val 점수가 가장 좋은 모델)를 드라이브에 저장하고 `train_log.csv`에 기록합니다.
**가장 오래 걸리는 셀**입니다(미측정). 세션이 끊기면 이 셀을 다시 실행하면 끊긴 에폭부터 이어 합니다.
`updates`를 쓰면 에폭 수는 자동으로 정해집니다(S0+S1 main20, 9,200번 갱신 = 67에폭). 에폭 하나가 S0의 약 3배 길어서, 한 에폭 시간은 약 3배가 됩니다(총 시간은 비슷할 것으로 예상, 미측정).
"""))
C.append(code("!python run.py train {ARGS}"))
C.append(md("""
## 6. eval, swaptest: val 점수
`eval`은 가장 좋은 모델로 val을 채점합니다. `swaptest`(영상+소리 모델만)는 같은 모델에 **진짜 영상 / 다른 클립 영상 / 0 영상**을 넣어 점수를 비교합니다(영상을 실제로 쓰는지 확인). 각각 수 분 이내.
소리만 쓰는 모델(B1)은 `swaptest`가 에러를 내는 것이 정상입니다.
"""))
C.append(code("""
!python run.py eval {ARGS}
if preset != "B1":
    !python run.py swaptest {ARGS}
"""))
C.append(md("""
## 7. predict: test 예측과 제출 파일
test 입력으로 예측 csv를 만들고 제출용 zip(`submission_*.zip`)으로 묶습니다. 수 분 이내. test 정답은 없으므로 점수는 나오지 않습니다(주최자가 채점).
"""))
C.append(code("!python run.py predict {ARGS}"))
C.append(md("""
## 8. summary: 결과 표 보기
지금까지 돌린 모든 실행을 `results/summary.csv` 한 표로 모으고, 같은 데이터 구성끼리 `results/gaps.csv`에 **"영상 덕분에 오른 점수"**(B2−B1, B2−(B2-0), B3−B1)를 계산합니다.
**`results/data_gaps.csv`는 추가 데이터의 효과**입니다: 같은 (데이터셋, 프리셋, 시드)에서 **기존 S0 실행 대비** S0+S1 등의 val 점수 차이(`F_ext_delta`, `F_basic_delta`)를 보여 줍니다. 비교하려면 같은 프리셋과 시드로 기존 실행(데이터 꼬리표 없음)이 있어야 합니다. 몇 초.
**주의:** 시드 하나의 차이(1%p 안팎)는 우연일 수 있습니다. 가능하면 시드를 2~3개로 반복하세요.
"""))
C.append(code("""
!python run.py summary {ARGS}
import pandas as pd
RES = f"{DRIVE_ROOT}/seld_runs/results"
display(pd.read_csv(f"{RES}/summary.csv"))
for name in ("gaps", "data_gaps"):
    if os.path.exists(f"{RES}/{name}.csv"):
        print(name); display(pd.read_csv(f"{RES}/{name}.csv"))
"""))
C.append(md("""
## 추천 실험 순서 (main20, S0 대 S0+S1)
모두 `dataset = main20`, `train_stages = 0,1`, `updates = 9200`, `seed = 1`로 두고 `preset`만 바꿔 실행합니다(기존 S0 실행과 비교됩니다).
1. `B1`, `B2`, `B2-0`, `B3` 네 개 (격차 `B2−B2-0` 등이 S0와 달라지는지 `gaps.csv`로 확인)
2. **갱신 횟수를 2배로**: `updates = 18400`으로 `B2` 한 번 (더 오래 학습하면 어떤지)
3. **요인 분해** (`B2`): `train_family = F1`(카메라 가족 효과), `train_mirror = 0`(거울 효과), `train_fraction = 0.5`(학습 곡선)
4. 시드 2, 3으로 반복
결과는 `data_gaps.csv`의 `F_ext_delta`, `F_basic_delta`로 봅니다.
"""))
from add_history import history_md                # 변경 이력 주석을 첫 셀 맨 위에 넣는다(이력 데이터는 add_history.py)
C[0]["source"] = (history_md("baseline/seld_challenge/colab_run.ipynb") + "\n\n" + "".join(C[0]["source"])).splitlines(True)
nb = {"cells": C, "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU", "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "..", "baseline", "seld_challenge", "colab_run.ipynb")
json.dump(nb, open(out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("저장:", os.path.abspath(out), len(C), "셀")
