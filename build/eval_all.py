# [변경 이력 시작]
#   2026-10-06  최초 생성: 8개 모델의 val, test 예측을 클립별 집계값으로 채점(부트스트랩 분석의 입력)
#   2026-10-06  드라이브 사본이 불완전한 ctrl5_B2 val 예측을 로컬 재생성본으로 대체, 일부 작업만 다시 돌리는 인자 추가
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""8개 모델(main20, ctrl5 × B1,B2,B2-0,B3)의 val·test 예측을 정답으로 채점하고, 클립별 집계값(TP, FP, FN, 각도 오차 합 등)을 저장한다.
이 값으로 analyze_results.py 가 같은 창 단위 부트스트랩을 한다. 점수 계산은 baseline/seld_challenge/metrics.py 의 SELDMetrics 를 그대로 쓴다.
python eval_all.py"""
import os, sys, csv, time
import numpy as np
from multiprocessing import Pool

CODE = r"G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset\baseline\seld_challenge"
sys.path.insert(0, CODE)
from metrics import SELDMetrics
from seld_label_utils import load_labels, organize_labels

RUNS = r"G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset\seld_runs\runs"
DS = r"C:\asrwork\dataset"
OUT = r"C:\asrwork\evalall"
LEN = {"main20": 200, "ctrl5": 50}
PRESETS = ["B1", "B2", "B2-0", "B3"]
KEYS = ["TP", "FP", "FPs", "FN", "Nref", "Ang", "Rel", "Dist", "Onsc", "DETP"]   # 클래스별(13) 합


def vec(m):
    return np.stack([m._TP, m._FP, m._FP_spatial, m._FN, m._Nref, m._total_AngE, m._total_RelDistE, m._total_DistE,
                     m._total_OnscreenCorrect, m._DE_TP])          # (10, 13)


def one(task):
    ds, p, split = task
    run = f"{ds}_{p}_s1"
    ref = os.path.join(DS, ds, "val", "labels") if split == "val" else os.path.join(DS, ds, "private", "test_labels")
    pred = os.path.join(RUNS, run, "pred_val_real" if split == "val" else "pred_test")
    alt = os.path.join("C:/asrwork/recheck", run, "pred_val_real")     # 드라이브 사본이 불완전해서 로컬에서 다시 만든 예측(ctrl5_B2 val)
    if split == "val" and os.path.isdir(alt) and len(os.listdir(pred)) < len(os.listdir(alt)):
        pred = alt
    names = sorted(f[:-4] for f in os.listdir(ref) if f.endswith(".csv"))
    n = LEN[ds]
    mk = lambda fold: SELDMetrics(doa_threshold=20, dist_threshold=np.inf, reldist_threshold=1.0, req_onscreen=False,
                                  nb_classes=13, average="macro", fold_azimuth=fold, exclude_absent_classes=True)
    arr = np.zeros((2, len(names), len(KEYS), 13))
    t0 = time.time(); miss = 0
    for i, nm in enumerate(names):
        gt = organize_labels(load_labels(os.path.join(ref, nm + ".csv"), convert_to_cartesian=False), n)
        pp = os.path.join(pred, nm + ".csv")
        if os.path.exists(pp):
            pr = organize_labels(load_labels(pp, convert_to_cartesian=False), n)
        else:
            pr = organize_labels({}, n); miss += 1
        for f in (0, 1):
            m = mk(bool(f)); m.update_seld_scores(pr, gt); arr[f, i] = vec(m)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, f"{run}_{split}.npz"), names=np.array(names), arr=arr, keys=np.array(KEYS))
    return run, split, len(names), miss, time.time() - t0


if __name__ == "__main__":
    tasks = [(ds, p, sp) for sp in ("test", "val") for ds in ("ctrl5", "main20") for p in PRESETS]
    if len(sys.argv) > 1:                       # 예: python eval_all.py ctrl5:B2:val  (일부만 다시)
        tasks = [tuple(a.split(":")) for a in sys.argv[1:]]
    with Pool(8) as pool:
        for r in pool.imap_unordered(one, tasks):
            print("done %s %s clips=%d missing_pred=%d %.0fs" % r, flush=True)
    print("ALLDONE")
