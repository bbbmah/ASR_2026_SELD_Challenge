# [변경 이력 시작]
#   2026-10-07  최초 생성: colab_generate.ipynb를 만드는 스크립트
#   2026-10-07  GPU 렌더러일 때 워커 수를 최대 6개로 제한(워커마다 약 2GB 메모리)
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""colab_generate.ipynb 를 만든다 (python build/make_generate_notebook.py)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
def code(t): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t.strip("\n").splitlines(True)}
C = []
C.append(md("""
# 추가 학습 데이터 생성 노트북 (코랩)

`DATA_PLAN_v2.md`의 데이터 확장(균일 격자 시간 창, 카메라 궤적 가족 F1~F5, 좌우 거울)을 코랩에서 만듭니다. **위에서부터 셀을 차례로 실행**하고, 2번 셀의 설정 칸만 바꾸면 됩니다.

**한 번에 하는 일**: 창(20초) 하나마다 main20 클립 1개와 ctrl5 조각 4개를 같이 만들고(같은 원본 프레임을 디코딩 한 번으로 공유), 250창씩 한 묶음으로 zip을 만들어 드라이브로 옮기고, **검증이 끝난 묶음만 로컬에서 지웁니다**(로컬 버퍼 방식). 중간에 끊겨도 같은 셀을 다시 실행하면 끝난 묶음은 건너뛰고 이어 합니다.

**첫 실행은 묶음 1개만** 만들도록 되어 있습니다(`MAX_SHARDS = 1`). 속도를 확인한 뒤 `0`(전부)으로 바꾸세요.

| 위치 | 내용 |
|---|---|
| 입력 (드라이브) | `STARSS23_datasets/{metadata_dev,foa_dev,video_dev}.zip` |
| 코드 (드라이브) | `build/gen_v2.py`, `build/dsgen.py` |
| 출력 (드라이브) | `generated/s{단계}/` : `plan.csv`, `status.json`, `s{단계}_main20_train_p000.zip`, `s{단계}_ctrl5_train_p000.zip`, `..._manifest_p000.csv`, 병합된 매니페스트 |
| 로컬 버퍼 | `/content/raw`(원천, 약 9GB), `/content/buf`(묶음 하나, 약 1.5GB) |

런타임은 **GPU**로 바꾸면 렌더링이 빨라집니다(창당 CPU 시간이 약 13초에서 약 4초로 줄 것으로 추정, 코랩에서는 첫 묶음으로 실측).
"""))
C.append(md("""
## 1. 드라이브 연결과 환경 확인
드라이브를 연결하고 GPU, CPU 수, 디스크, 메모리를 보여 줍니다. 1분 이내.
`DRIVE_ROOT`는 `build/`, `STARSS23_datasets/`가 들어 있는 프로젝트 폴더입니다. 다르면 고치세요.
"""))
C.append(code("""
from google.colab import drive
drive.mount('/content/drive')

DRIVE_ROOT = "/content/drive/MyDrive/Colab Notebooks/ASR_2026-2/for_dataset"   #@param {type:"string"}

import os, sys
assert os.path.isdir(DRIVE_ROOT + "/build"), "DRIVE_ROOT 경로를 확인하세요 (build 폴더가 있어야 합니다)"
!pip -q install soundfile
!nvidia-smi -L
!nproc
!df -h /content | tail -1
!free -g | head -2
"""))
C.append(md("""
## 2. 설정 칸 (여기만 바꾸세요)
- `STAGE`: 단계 번호. 1단계는 약 6,000창을 만듭니다. 2단계 이상은 **다른 새 창**(다른 시드)을 만듭니다.
- `N_WINDOWS`: 이 단계에서 만들 창 수(= main20 클립 수). ctrl5 조각은 이것의 4배.
- `SHARD_WINDOWS`: 묶음 하나의 창 수(기본 250창 = 약 1.5GB).
- `MAX_SHARDS`: 이번 실행에서 처리할 묶음 수. **0이면 남은 전부**. 처음에는 1.
- `RENDERER`: `auto`(GPU가 있고 검증을 통과하면 GPU), `gpu`, `cpu`.
- `WORKERS`: 병렬 프로세스 수. 0이면 자동(CPU 렌더러는 CPU 수, GPU 렌더러는 최대 6개: 워커마다 메모리를 약 2GB 쓰므로 램이 12GB인 일반 런타임에서는 4 이하로 줄이는 것이 안전합니다).
- `MIRROR_FRACTION`: 좌우 거울로 만들 창의 비율.
- `FAMILY_MIX`: 카메라 궤적 가족 비율. F1 = 챌린지 정의(sweep), F2 느림, F3 빠름, F4 정지-회전, F5 왕복.
- `FULL_VERIFY`: 드라이브에 올린 zip을 처음부터 끝까지 다시 읽어 검사합니다(권장, 묶음당 1~2분).

**단계, 창 수, 비율을 바꾸면 계획이 달라집니다.** 이미 계획이 있는 폴더에서는 오류로 알려 주니, 새로 만들려면 `STAGE`를 바꾸세요.
"""))
C.append(code("""
STAGE = 1                      #@param {type:"integer"}
N_WINDOWS = 6000               #@param {type:"integer"}
SHARD_WINDOWS = 250            #@param {type:"integer"}
MAX_SHARDS = 1                 #@param {type:"integer"}
RENDERER = "auto"              #@param ["auto", "gpu", "cpu"]
WORKERS = 0                    #@param {type:"integer"}
MIRROR_FRACTION = 0.5          #@param {type:"number"}
RARE_BONUS = True              #@param {type:"boolean"}
FAMILY_MIX = "F1=0.5,F2=0.1,F3=0.1,F4=0.15,F5=0.15"   #@param {type:"string"}
FULL_VERIFY = True             #@param {type:"boolean"}

RAW_DIR = "/content/raw"
BUF_DIR = "/content/buf"
OUT_DIR = f"{DRIVE_ROOT}/generated/s{STAGE}"
print("출력 폴더:", OUT_DIR)
"""))
C.append(md("""
## 3. 원천 데이터 풀기
드라이브의 STARSS23 zip(약 11GB)을 코랩 로컬 `/content/raw`에 풉니다. 이미 풀려 있으면 건너뜁니다. 세션이 새로 시작되면 처음 한 번 **약 5~10분** 걸립니다.
끝나면 `{'metadata_dev': 168, 'foa_dev': 168, 'video_dev': 156}`이 나와야 합니다.
"""))
C.append(code("""
import glob
if not os.path.exists(RAW_DIR + "/.done"):
    os.makedirs(RAW_DIR, exist_ok=True)
    for z in ("metadata_dev", "foa_dev", "video_dev"):
        zp = f"{DRIVE_ROOT}/STARSS23_datasets/{z}.zip"
        assert os.path.exists(zp), zp
        !unzip -q -o "{zp}" -d {RAW_DIR}
    open(RAW_DIR + "/.done", "w").write("ok")
print({k: len(glob.glob(f"{RAW_DIR}/{k}/*/*")) for k in ("metadata_dev", "foa_dev", "video_dev")})
"""))
C.append(md("""
## 4. 코드 불러오기
드라이브의 생성 코드를 불러옵니다. (원천 위치 `RAW`는 불러오기 **전에** 정해야 하므로, 바꾸려면 런타임을 다시 시작하세요.)
"""))
C.append(code("""
os.environ["RAW"] = RAW_DIR
sys.path.insert(0, DRIVE_ROOT + "/build")
import importlib, json, time, math
import numpy as np, pandas as pd, torch
import dsgen as G
import gen_v2 as V
importlib.reload(V)
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "| CPU", os.cpu_count())
"""))
C.append(md("""
## 5. 창 계획 만들기
train 방 녹음에서 **균일 격자**로 창을 뽑고, 창마다 카메라 가족, 거울 여부, ctrl5 조각의 고정 요 4개를 정합니다. 결과는 드라이브 `plan.csv`에 저장되어 **다음 실행에서도 같은 계획을 씁니다.** 아무것도 렌더링하지 않습니다(몇 초~1분).
요약 표에서 확인할 것: 창 수가 목표(`N_WINDOWS`) 근처인지, 가족 비율, 거울 비율(약 50%), 시간 커버 배수(1단계 약 11배 = 기존 5.4배와 합쳐 약 16배).
"""))
C.append(code("""
mix = {k: float(v) for k, v in (x.split("=") for x in FAMILY_MIX.split(","))}
valid = V.load_valid()
split_of, info = G.make_split(valid)
vb = {p["name"]: p for p in valid}
print("val 방", info["val_rooms"], "| train 방", info["train_rooms"], "| test 방", info["test_rooms"], "(확장은 train 방에서만)")

os.makedirs(OUT_DIR, exist_ok=True)
sig = dict(stage=STAGE, n=N_WINDOWS, mix=mix, mirror=MIRROR_FRACTION, rare=RARE_BONUS, seed=V.SEED)
pp, mp = OUT_DIR + "/plan.csv", OUT_DIR + "/plan_params.json"
if os.path.exists(pp):
    old = json.load(open(mp))
    assert old == sig, f"기존 계획과 설정이 다릅니다. 이어 하려면 설정을 되돌리고, 새로 만들려면 STAGE 를 바꾸세요.\\n기존: {old}\\n지금: {sig}"
    plan = pd.read_csv(pp)
    print("기존 계획 사용:", len(plan), "개 창")
else:
    plan, meta = V.make_plan(valid, split_of, STAGE, N_WINDOWS, mix, MIRROR_FRACTION, RARE_BONUS)
    plan.to_csv(pp, index=False); json.dump(sig, open(mp, "w"))
    print("새 계획 저장:", len(plan), "개 창")
display(pd.Series(V.plan_summary(plan, valid, split_of)))
"""))
C.append(md("""
## 6. 자체 점검과 렌더러 선택
생성 전에 꼭 통과해야 하는 점검입니다. 약 1~2분.
1. **거울**: 거울 오디오 = 원래 오디오의 좌우 교환, 거울 영상 = 원래 렌더링의 좌우 뒤집기, 거울 라벨의 방위각 부호 반전과 `onscreen` 불변
2. **카메라 가족**: 각속도 45°/s 이하, 피치 −25~0°, 빠른 회전에서도 소리가 튀지 않음
3. **GPU 렌더러**가 기존 CPU 렌더러와 같은 영상을 만드는지(평균 차이 < 1/255)
4. 창 2개를 실제로 만들어 프레임 수, 샘플 수, 라벨 행 수가 맞는지

**`PASS: true`가 아니면 여기서 멈춥니다.**
"""))
C.append(code("""
dev = "cuda" if torch.cuda.is_available() else "cpu"
res = V.selftest(valid, split_of, gpu_device=dev)
assert res["PASS"], "자체 점검 실패: 위 결과를 확인하세요"

d = res.get("gpu_vs_cpu_mean_abs_diff")
if RENDERER == "auto":
    renderer = "gpu" if (dev == "cuda" and d is not None and d < 1.0) else "cpu"
else:
    renderer = RENDERER
ncpu = os.cpu_count() or 2
workers = WORKERS or (min(ncpu, 6) if renderer == "gpu" else ncpu)    # GPU 렌더러는 워커마다 약 2GB 메모리를 쓰므로 6개로 제한
print(f"\\n선택: 렌더러 = {renderer}, 워커 = {workers}  (GPU vs CPU 렌더 차이 {d})")
"""))
C.append(md("""
## 7. 생성 (묶음 단위, 이어 하기 가능)
묶음마다 **생성 → zip → 드라이브 복사 → 검증 → 로컬 삭제**를 합니다. 검증(크기, zip 목록과 CRC, 전체 읽기)이 모두 통과한 묶음만 로컬에서 지웁니다. 문제가 있으면 그 묶음의 로컬 파일을 남기고 멈춥니다.

- 처음에는 `MAX_SHARDS = 1`이라 묶음 1개(약 250창)만 만듭니다. 끝나면 **묶음당 시간과 남은 시간 추정**이 나옵니다. 괜찮으면 2번 셀에서 `MAX_SHARDS = 0`으로 바꾸고 이 셀을 다시 실행하세요(끝난 묶음은 건너뜁니다).
- **오래 걸리는 셀**입니다(시간은 코랩 첫 묶음으로 실측). 세션이 끊기면 3~7번 셀을 순서대로 다시 실행하면 이어서 합니다.
- 소리가 너무 커서(피크 0.9999 초과) 버려진 창은 `status.json`의 `dropped_peak`에 기록됩니다. 그 창은 main20과 ctrl5 양쪽에서 빠집니다.
"""))
C.append(code("""
t0 = time.time()
status = V.run_generation(plan, vb, STAGE, OUT_DIR, BUF_DIR, workers, renderer,
                          shard_windows=SHARD_WINDOWS, max_shards=MAX_SHARDS, full_verify=FULL_VERIFY)
print(f"\\n이번 실행 {(time.time() - t0) / 60:.1f}분")
done = [v for v in status.values() if v["ok"]]
if done:
    per = float(np.mean([v["secs"] for v in done])); n_sh = math.ceil(len(plan) / SHARD_WINDOWS); left = n_sh - len(done)
    print(f"끝난 묶음 {len(done)}/{n_sh}, 묶음당 평균 {per:.0f}초 → 남은 {left}개 약 {left * per / 3600:.1f}시간 (창당 {per / SHARD_WINDOWS:.1f}초)")
"""))
C.append(md("""
## 8. 결과 요약과 매니페스트 병합
지금까지 만든 묶음의 개수, 용량, 버려진 창, 문제를 보여 주고, 묶음별 매니페스트를 하나로 합쳐 저장합니다(`s{단계}_main20_manifest.csv`, `s{단계}_ctrl5_manifest.csv`). 몇 초.
"""))
C.append(code("""
st = json.load(open(OUT_DIR + "/status.json"))
ok = [k for k, v in st.items() if v["ok"]]; bad = [k for k, v in st.items() if not v["ok"]]
gen = sum(st[k]["generated"] for k in ok); drop = sum(len(st[k]["dropped_peak"]) for k in ok)
gb = sum(os.path.getsize(f) for f in glob.glob(OUT_DIR + "/*.zip")) / 2**30
print(f"정상 묶음 {len(ok)}개 | 생성된 창 {gen}개 (main20 클립 {gen}개 + ctrl5 조각 {gen * 4}개) | 피크로 버린 창 {drop}개 | zip {gb:.1f}GB")
if bad: print("문제 있는 묶음:", {k: (st[k]["failed"], [r["ok"] for r in st[k]["zips"]]) for k in bad})
print("매니페스트 병합:", V.merge_manifests(OUT_DIR, STAGE))
print("남은 묶음:", math.ceil(len(plan) / SHARD_WINDOWS) - len(ok))
"""))
C.append(md("""
## 9. 마무리 (선택)
- 다 끝났으면 드라이브 쓰기를 확실히 밀어내고 연결을 끊습니다. **클라우드 동기화가 완전히 끝났는지는 PC의 드라이브 데스크톱 상태나 웹에서 확인**하세요(코랩의 쓰기 지연 때문).
- `/content/buf`, `/content/raw`는 세션이 끝나면 사라지지만, 이어서 다른 작업을 하려면 아래 셀로 지울 수 있습니다.
"""))
C.append(code("""
import shutil
shutil.rmtree(BUF_DIR, ignore_errors=True)       # 로컬 버퍼 (검증된 묶음은 이미 지워져 있음)
# shutil.rmtree(RAW_DIR, ignore_errors=True)      # 원천까지 지우려면 주석을 푸세요 (다시 풀면 5~10분)
drive.flush_and_unmount()
"""))
from add_history import history_md                # 변경 이력 주석을 첫 셀 맨 위에 넣는다(이력 데이터는 add_history.py)
C[0]["source"] = (history_md("colab_generate.ipynb") + "\n\n" + "".join(C[0]["source"])).splitlines(True)
nb = {"cells": C, "metadata": {"colab": {"provenance": [], "gpuType": "T4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}, "accelerator": "GPU", "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(HERE, "..", "colab_generate.ipynb")
json.dump(nb, open(out, "w", encoding="utf8"), ensure_ascii=False, indent=1)
print("저장:", os.path.abspath(out), len(C), "셀")
