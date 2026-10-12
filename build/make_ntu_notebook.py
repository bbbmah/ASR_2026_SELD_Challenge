# [변경 이력 시작]
#   2026-10-13  최초 생성: baseline/ntu_challenge/colab_ntu.ipynb를 만드는 스크립트
# [변경 이력 끝]
"""baseline/ntu_challenge/colab_ntu.ipynb (NTU 시스템 학습 노트북)를 만든다. python build/make_ntu_notebook.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}
C = []
C.append(md("""
# NTU 시스템 학습 노트북 (챌린지 베이스라인 중 하나)

DCASE 2025 참가팀 NTU의 소리만 쓰는 시스템을 챌린지 데이터셋(`main20`, `ctrl5`)으로 **학습**합니다. 설명은 `README_ntu.md`를 보세요.
위에서부터 **셀을 순서대로** 실행하고 2번 셀의 **설정 칸**만 바꾸면 됩니다. 런타임은 **GPU**(런타임 → 런타임 유형 변경)로 설정해 주세요.

- 하는 일: zip 풀기 → 특징 뽑기(드라이브에 캐시) → 학습 → 매 에폭 val 채점으로 최고 모델 저장.
- 하지 않는 일: test 예측과 제출 파일 만들기, 이어 하기. **학습 중 세션이 끊기면 학습을 처음부터 다시 해야 합니다**(특징 캐시는 드라이브에 남아 있어서 다시 뽑지 않습니다).
- 소리만 쓰므로 영상은 쓰지 않습니다. 결과는 `seld_runs/runs/ntu_{dataset}_s{seed}/` 에 저장되어 기존 `seld_challenge` 실행과 이름이 겹치지 않습니다.
"""))
C.append(md("""
## 1. 드라이브 연결, 폴더 위치, 패키지 설치
`DRIVE_ROOT`는 `dataset/`, `internal/`, `baseline/`이 들어 있는 프로젝트 폴더 경로입니다. 다르면 고치세요. 1분 이내.
"""))
C.append(code("""
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = "/content/drive/MyDrive/Colab Notebooks/ASR_2026-2/for_dataset"   #@param {type:"string"}
CODE_DIR = f"{DRIVE_ROOT}/baseline/ntu_challenge"

import os
os.chdir(CODE_DIR)
!pip install -q librosa soundfile scipy tqdm
!nvidia-smi -L
!free -g | head -2
!df -h /content | tail -1
"""))
C.append(md("""
## 2. 설정 칸 (여기만 바꾸세요)
- `dataset`: `main20`(카메라가 도는 20초, 라벨 200프레임) 또는 `ctrl5`(카메라 고정 5초, 50프레임).
- `nb_epochs`: NTU 원본 기본값은 100입니다(OneCycleLR이라 에폭 수에 맞춰 학습률이 오르내립니다. 도중에 멈추면 마지막 구간이 빠집니다).
- `batch_size`: 한 번에 GPU에 올리는 클립 수. GPU 메모리가 모자라면 줄이세요(`accum_batch`의 약수여야 합니다). `accum_batch`는 가중치를 한 번 갱신하는 클립 수로, `auto`는 main20 64, ctrl5 256입니다.
- `select_metric`: 가장 좋은 모델을 고르는 val 점수. `F_ext`(방위각을 접지 않음, 챌린지 점수) 또는 `F_basic`(접음, DCASE 공식).
- 특징 (`ms`, `gamma`, `iv`): NTU가 제안한 입력 특징. 켜면 입력 채널이 늘어 **특징 메모리가 커집니다**(main20 train: 기본 약 2.5GB, 셋 다 켜면 약 7.5GB). 특징은 조합마다 따로 캐시됩니다.
- `augmentation`: NTU가 제안한 증강. `none`이면 쓰지 않습니다.
- `exp`: 실행 이름 꼬리표(같은 설정을 다른 이름으로 따로 저장하고 싶을 때).
"""))
C.append(code("""
dataset = "main20"           #@param ["main20", "ctrl5"]
seed = 1                     #@param {type:"integer"}
nb_epochs = 100              #@param {type:"integer"}
batch_size = 32              #@param {type:"integer"}
accum_batch = "auto"         #@param {type:"string"}
learning_rate = 0.001        #@param {type:"number"}
val_freq = 2                 #@param {type:"integer"}
select_metric = "F_ext"      #@param ["F_ext", "F_basic"]
ms = False                   #@param {type:"boolean"}
gamma = False                #@param {type:"boolean"}
iv = False                   #@param {type:"boolean"}
augmentation = "none"        #@param ["none", "cutout", "itfm", "freqshift", "filtaug", "compfreq", "alltrans"]
exp = ""                     #@param {type:"string"}

import shlex
DATA_ROOT = "/content/data"                                          # zip 을 푸는 로컬 디스크
OUT_ROOT = f"{DRIVE_ROOT}/seld_runs"                                  # 결과와 특징 캐시 (드라이브)
ZIP_DIR = f"{DRIVE_ROOT}/dataset/main20" if dataset == "main20" else f"{DRIVE_ROOT}/internal"
_a = ["--dataset", dataset, "--data_root", DATA_ROOT, "--out_root", OUT_ROOT, "--zip_dir", ZIP_DIR, "--local_tmp", "/content/tmp",
      "--seed", seed, "--nb_epochs", nb_epochs, "--batch_size", batch_size, "--accum_batch", accum_batch,
      "--learning_rate", learning_rate, "--val_freq", val_freq, "--select_metric", select_metric]
for _name, _on in (("ms", ms), ("gamma", gamma), ("iv", iv)):
    if _on: _a.append("--" + _name)
if augmentation != "none": _a.append("--" + augmentation)
if exp: _a += ["--exp", exp]
ARGS = shlex.join(str(x) for x in _a)
print(ARGS)
"""))
C.append(md("""
## 3. 학습 (zip 풀기 → 특징 → 학습)
한 번에 합니다. 순서와 걸리는 시간(모두 **미측정**):
1. **zip 풀기**: 드라이브의 `train`, `val` zip을 `/content/data`에 풉니다(main20 train zip은 영상까지 들어 있어 약 11GB, 수 분). 이미 풀려 있으면 건너뜁니다.
2. **특징 뽑기**: 오디오만 뽑아 드라이브(`seld_runs/features/{dataset}/ntu_...`)에 500클립씩 묶어 저장합니다. 이미 있으면 건너뛰어서 **처음 한 번만 오래 걸립니다**.
3. **학습**: 매 `val_freq` 에폭(마지막 20%는 매 에폭) val을 채점해 가장 좋은 모델을 `best_model.pth`로 저장하고, `train_log.csv`에 기록합니다.
로그에 `에폭 n/N | 학습손실 | val손실 | F_ext F_basic | 최고` 가 나옵니다.
"""))
C.append(code("!python main.py {ARGS}"))
C.append(md("""
## 4. 결과 보기
이 실행의 최고 모델과 val 점수, 에폭별 기록 끝부분을 보여 줍니다. 몇 초.
"""))
C.append(code("""
import json, pandas as pd
RUN = f"{OUT_ROOT}/runs/ntu_{dataset}_s{seed}" + (f"_{exp}" if exp else "")
s = json.load(open(f"{RUN}/train_summary.json", encoding="utf8"))
v = s["val_scores"]
print(f"최고 에폭 {s['best_epoch']} / {s['epochs_done']}   {s['select_metric']} = {s['best_value']:.4f}   (갱신 {s['total_updates']}번, 학습 클립 {s['train_clips']}개)")
for k in ("ext", "basic"):
    r = v[k]; print(f"[{k}] F={r['F']:.4f}  DOA오차={r['DOA_err']:.2f}  거리오차(상대)={r['rel_dist_err']:.3f}")
print("모델:", f"{RUN}/best_model.pth")
display(pd.read_csv(f"{RUN}/train_log.csv").dropna(subset=["F_ext"]).tail(10))
"""))
from add_history import history_md                # 변경 이력 주석을 첫 셀 맨 위에 넣는다(이력 데이터는 add_history.py)
C[0]["source"] = (history_md("baseline/ntu_challenge/colab_ntu.ipynb") + "\n\n" + "".join(C[0]["source"])).splitlines(True)
nb = {"cells": C, "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU", "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "..", "baseline", "ntu_challenge", "colab_ntu.ipynb")
json.dump(nb, open(out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
