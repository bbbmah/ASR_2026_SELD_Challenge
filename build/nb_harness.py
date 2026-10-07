# [변경 이력 시작]
#   2026-10-07  최초 생성: 노트북 셀 코드를 코랩 없이(드라이브 연결, 원천 풀기 제외) 실행해 보는 하네스
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
import json, re, os, sys, shutil
NB = r"G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset\colab_generate.ipynb"
def strip(src): return "\n".join(re.sub(r"^(\s*)[!%].*$", r"\1pass", l) for l in src.split("\n"))
if __name__ == "__main__":
    base = "C:/asrwork/nb_test"; shutil.rmtree(base, ignore_errors=True); os.makedirs(base)
    cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf8"))["cells"] if c["cell_type"] == "code"]
    ns = {"display": print, "__name__": "__main__"}
    ns["DRIVE_ROOT"] = r"G:/내 드라이브/Colab Notebooks/ASR_2026-2/for_dataset"
    exec(strip(cells[1]), ns)                                     # 설정 칸
    ns.update(DRIVE_ROOT=r"G:/내 드라이브/Colab Notebooks/ASR_2026-2/for_dataset", RAW_DIR="C:/asrwork/raw", BUF_DIR=base + "/buf",
              STAGE=1, N_WINDOWS=6, SHARD_WINDOWS=3, MAX_SHARDS=1, WORKERS=2, FULL_VERIFY=True)
    ns["OUT_DIR"] = base + "/drive/generated/s1"; os.makedirs(ns["OUT_DIR"], exist_ok=True)
    ns["os"] = os; ns["sys"] = sys
    for i in (3, 4, 5, 6):                                         # 불러오기, 계획, 점검, 생성 (0: 드라이브, 2: 원천 풀기는 건너뜀)
        print(f"\n######## 셀 {i} ########", flush=True)
        exec(strip(cells[i]) if i != 6 else strip(cells[6]), ns)
    # 이어 하기 (설정 같음)
    ns["MAX_SHARDS"] = 0
    print("\n######## 셀 6 다시 (이어 하기) ########", flush=True); exec(strip(cells[6]), ns)
    ns["glob"] = __import__("glob"); ns["math"] = __import__("math")
    print("\n######## 셀 7 (요약) ########", flush=True); exec(strip(cells[7]), ns)
