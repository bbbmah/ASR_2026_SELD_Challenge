# [변경 이력 시작]
#   2026-10-03  최초 생성: 채점 시험 3개(정답 그대로, 앞뒤 뒤집기, 빈 클립에 가짜 예측)
#   2026-10-03  (c) 시험의 출력을 소수 여섯째 자리까지 정확히 표시하도록 수정
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""채점 시험 3개 (val 정답 사용). 실제 학습 없이 score.py 가 맞게 동작하는지만 본다.
python check_scoring.py <데이터셋 루트(dataset 폴더)>"""
import os, sys, glob, shutil, tempfile
import pandas as pd
from score import score_dirs, one_line

root = sys.argv[1] if len(sys.argv) > 1 else r"C:\asrwork\dataset"
HDR = ["frame", "class", "source", "azimuth", "distance", "onscreen"]


def wrap(a): return (a + 180) % 360 - 180


def make_pred(ref_dir, out, fn=None):
    os.makedirs(out, exist_ok=True)
    for f in glob.glob(os.path.join(ref_dir, "*.csv")):
        d = pd.read_csv(f)
        if fn: d = fn(os.path.basename(f), d)
        d.to_csv(os.path.join(out, os.path.basename(f)), index=False, columns=HDR)


tmp = tempfile.mkdtemp()
for ds, n in (("main20", 200), ("ctrl5", 50)):
    ref = os.path.join(root, ds, "val", "labels")
    print(f"===== {ds} val ({len(glob.glob(os.path.join(ref, '*.csv')))}개 클립, 길이 {n}) =====")
    # (a) 정답 = 예측
    pa = os.path.join(tmp, ds + "_a"); make_pred(ref, pa)
    ra = score_dirs(pa, ref, n); print("(a) 정답 그대로:", one_line(ra))
    ok_a = all(abs(ra[k]["F"] - 1) < 1e-9 and abs(ra[k]["DOA_err"]) < 1e-9 for k in ("ext", "basic"))
    # (b) 앞뒤 뒤집기: az -> 180 - az
    def flip(name, d): d = d.copy(); d["azimuth"] = wrap(180 - d["azimuth"]).round().astype(int); return d
    pb = os.path.join(tmp, ds + "_b"); make_pred(ref, pb, flip)
    rb = score_dirs(pb, ref, n); print("(b) 앞뒤 뒤집기:", one_line(rb))
    ok_b = abs(rb["basic"]["F"] - 1) < 1e-9 and rb["ext"]["F"] < 0.9
    # (c) 정답이 빈 클립에 가짜 예측 1개
    empties = [os.path.basename(f) for f in glob.glob(os.path.join(ref, "*.csv")) if len(pd.read_csv(f)) == 0]
    if empties:
        target = empties[0]
        pc = os.path.join(tmp, ds + "_c")
        def addfake(name, d):
            if name == target: d = pd.DataFrame([[10, 0, 0, 0, 100, 0]], columns=HDR)
            return d
        make_pred(ref, pc, addfake)
        rc = score_dirs(pc, ref, n); print(f"(c) 빈 클립({target})에 가짜 예측 1개(class 0): F_ext={rc['ext']['F']:.6f}, F_basic={rc['basic']['F']:.6f} (정답 그대로면 1.000000)")
        ok_c = rc["ext"]["F"] < 1 - 1e-9 and rc["basic"]["F"] < 1 - 1e-9
    else:
        print("(c) 이 데이터셋 val 에는 정답이 빈 클립이 없음 -> ctrl5 로 확인"); ok_c = None
    print(f"   결과: (a) {'통과' if ok_a else '실패'} | (b) {'통과' if ok_b else '실패'} | (c) {'통과' if ok_c else ('해당없음' if ok_c is None else '실패')}")
    # 참고: 정답 없는 클래스를 평균에 넣으면 F 가 1 보다 작아진다
    rk = score_dirs(pa, ref, n, exclude_absent_classes=False); print("   (참고) 정답 없는 클래스 포함 시 F_ext =", round(rk["ext"]["F"], 3))
shutil.rmtree(tmp)
