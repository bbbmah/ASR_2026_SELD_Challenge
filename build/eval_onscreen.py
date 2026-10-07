# [변경 이력 시작]
#   2026-10-06  최초 생성: 영상+소리 모델을 onscreen까지 맞아야 정답인 조건(공식 F(20/1/on))으로 다시 채점
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""영상+소리 모델(B2, B2-0, B3)을 'onscreen 정확히 맞혀야 TP' 조건(공식 F(20°/1/on))으로 다시 채점한다.
python eval_onscreen.py"""
import os, sys, json, time
from multiprocessing import Pool
sys.path.insert(0, r"G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset\baseline\seld_challenge")
from score import score_dirs
RUNS = r"G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset\seld_runs\runs"
DS = r"C:\asrwork\dataset"; LEN = {"main20": 200, "ctrl5": 50}

def one(t):
    ds, p, sp = t
    run = f"{ds}_{p}_s1"
    ref = os.path.join(DS, ds, "val", "labels") if sp == "val" else os.path.join(DS, ds, "private", "test_labels")
    pred = os.path.join(RUNS, run, "pred_val_real" if sp == "val" else "pred_test")
    alt = os.path.join("C:/asrwork/recheck", run, "pred_val_real")
    if sp == "val" and os.path.isdir(alt) and len(os.listdir(pred)) < len(os.listdir(alt)): pred = alt
    t0 = time.time()
    r = score_dirs(pred, ref, LEN[ds], "audio_visual", None, exclude_absent_classes=True, lad_req_onscreen=True)
    json.dump(r, open(rf"C:\asrwork\evalall\onscreen_{run}_{sp}.json", "w"))
    return run, sp, r["ext"]["F"], r["basic"]["F"], time.time() - t0

if __name__ == "__main__":
    tasks = [(ds, p, sp) for sp in ("test", "val") for ds in ("ctrl5", "main20") for p in ("B2", "B2-0", "B3")]
    with Pool(8) as pool:
        for r in pool.imap_unordered(one, tasks): print("done %s %s F_ext_on=%.4f F_basic_on=%.4f %.0fs" % r, flush=True)
    print("ALLDONE")
