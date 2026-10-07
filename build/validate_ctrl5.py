# [변경 이력 시작]
#   2026-10-03  최초 생성: ctrl5 v2 필수 확인 5개(개수, 같은 구간, 빈 라벨 읽기, 채점 단위, 공개 묶음 점검)
#   2026-10-03  m.clip 오류를 ["clip"]으로 수정
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""ctrl5 v2 필수 확인 5개. python validate_ctrl5.py [1 2 3 4 5]"""
import sys, os, glob, zipfile, tempfile, types, importlib
from unittest import mock
import numpy as np, pandas as pd

DS = r"C:\asrwork\dataset"
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "..", "baseline", "DCASE2025_seld_baseline")
SPL = ("train", "val", "test")


def lab_dir(ds, sp): return os.path.join(DS, ds, "private", "test_labels") if sp == "test" else os.path.join(DS, ds, sp, "labels")
def meta_dir(ds, sp): return os.path.join(DS, ds, "private", "test_meta") if sp == "test" else os.path.join(DS, ds, sp, "meta")


def v1():
    m = pd.read_csv(os.path.join(DS, "main20", "manifest.csv"))
    ok = True
    for sp in SPL:
        rem = int(((m.split == sp) & (m.dropped == 0)).sum())
        on_disk = len(glob.glob(os.path.join(lab_dir("main20", sp), "*.csv")))
        c5 = len(glob.glob(os.path.join(lab_dir("ctrl5", sp), "*.csv")))
        vids = len(glob.glob(os.path.join(DS, "ctrl5", sp, "video", "*.mp4"))); auds = len(glob.glob(os.path.join(DS, "ctrl5", sp, "audio", "*.wav")))
        good = (rem == on_disk) and (c5 == 4 * rem) and (vids == c5 == auds)
        ok &= good
        print(f"{sp}: main20 남은 {rem} (디스크 {on_disk}) -> ctrl5 {c5} (기대 {4*rem}), video {vids}, audio {auds}  {'OK' if good else 'FAIL'}")
    print("V1", "PASS" if ok else "FAIL")


def classcount(ds, sp):
    c = np.zeros(13, int); n = 0
    for f in glob.glob(os.path.join(lab_dir(ds, sp), "*.csv")):
        d = pd.read_csv(f); c += np.bincount(d["class"], minlength=13)[:13]; n += len(d)
    return c, n


def v2():
    ok = True
    for sp in SPL:
        a, na = classcount("main20", sp); b, nb = classcount("ctrl5", sp)
        same = (a == b).all(); ok &= same
        print(f"{sp}: 클래스별 행 수 main20 == ctrl5 : {same} (총 {na} vs {nb})")
        if not same: print("  main20", a.tolist(), "\n  ctrl5 ", b.tolist())
    pm = pd.read_csv(os.path.join(DS, "ctrl5", "private", "parent_map.csv"))
    cols = ["frame", "class", "source", "az_world", "el_world", "distance"]
    bad = 0; checked = 0; cache = {}
    for sp in SPL:
        sub = pm[pm["clip"].str.startswith(sp + "_")]
        for r in sub.itertuples():
            if r.parent_clip not in cache:
                cache = {r.parent_clip: pd.read_csv(os.path.join(meta_dir("main20", sp), r.parent_clip + "_ext.csv"))}
            pe = cache[r.parent_clip]
            pe = pe[(pe.frame >= 50 * r.j) & (pe.frame < 50 * r.j + 50)][cols].reset_index(drop=True)
            ce = pd.read_csv(os.path.join(meta_dir("ctrl5", sp), r.clip + "_ext.csv"))[cols].copy()
            ce["frame"] = ce["frame"] + 50 * r.j
            checked += 1
            if len(pe) != len(ce) or not (pe.values == ce.reset_index(drop=True).values).all():
                bad += 1
                if bad <= 5: print("  불일치:", r.clip, r.parent_clip, r.j, len(pe), len(ce))
    print(f"조각 {checked}개 meta 비교, 불일치 {bad}")
    ok &= bad == 0
    print("V2", "PASS" if ok else "FAIL")


def v3():
    # torch 등이 없어도 베이스라인 utils/metrics를 import 하도록 무거운 모듈만 가짜로 대체
    for name in ["torch", "torch.utils", "torch.utils.tensorboard", "librosa", "librosa.feature", "PIL", "PIL.Image"]:
        sys.modules.setdefault(name, mock.MagicMock())
    sys.path.insert(0, BASE)
    import utils as U, metrics as M
    tmp = tempfile.mkdtemp(); ref = os.path.join(tmp, "ref", "sub"); pred = os.path.join(tmp, "pred"); os.makedirs(ref); os.makedirs(pred)
    hdr = "frame,class,source,azimuth,distance,onscreen\n"
    open(os.path.join(ref, "empty.csv"), "w").write(hdr)
    open(os.path.join(ref, "full.csv"), "w").write(hdr + "3,1,1,40,200,1\n4,1,1,42,200,1\n")
    # (a) 읽기
    d = U.load_labels(os.path.join(ref, "empty.csv")); print("load_labels(empty) ->", d)
    print("organize_labels(empty, 0) ->", U.organize_labels(d, 0))
    # (b) 평가: 정답이 빈 파일에 예측이 있는 경우와 정답 파일 끝 뒤쪽 프레임에 예측이 있는 경우
    open(os.path.join(pred, "empty.csv"), "w").write(hdr + "0,1,0,10,100,0\n1,1,0,10,100,0\n")
    open(os.path.join(pred, "full.csv"), "w").write(hdr + "3,1,0,40,200,1\n4,1,0,42,200,1\n30,5,0,0,100,0\n")
    params = dict(root_dir=ref, lad_doa_thresh=20, lad_dist_thresh=np.inf, lad_reldist_thresh=1.0, lad_req_onscreen=False,
                  modality="audio_visual", average="micro", nb_classes=13)
    res = M.ComputeSELDResults(params, ref_files_folder=os.path.join(tmp, "ref"))
    F, AngE, DistE, RelDistE, On, _ = res.get_SELD_Results(pred)
    print(f"크래시 없음. micro F={F:.3f} AngE={AngE:.2f}")
    sm = M.SELDMetrics(nb_classes=13, average="micro")
    pe = U.organize_labels(U.load_labels(os.path.join(pred, "empty.csv"), convert_to_cartesian=False), 2)
    ge = U.organize_labels(U.load_labels(os.path.join(ref, "empty.csv"), convert_to_cartesian=False), 0)
    sm.update_seld_scores(pe, ge)
    print(f"빈 정답 + 예측 2건 -> FP={sm._FP.sum():.0f} (0이면 빈 구간의 오검출을 세지 않음)")
    print("V3 PASS (읽기/평가 모두 크래시 없음; 단, 위 FP 값 참고)")


def v4():
    print("metrics.py 라인 근거는 NOTES.md에 기록 (코드 읽기).")
    src = open(os.path.join(BASE, "metrics.py"), encoding="utf8").read().splitlines()
    for ln in (120, 122, 124, 128, 169, 177, 223, 224, 246, 249):
        print(ln, src[ln - 1].strip())


def v5():
    ok = True
    for sp in ("train.zip", "val.zip", "test_inputs.zip", "private_test.zip"):
        names = zipfile.ZipFile(os.path.join(DS, "main20", sp)).namelist()
        if sp != "private_test.zip":
            bad = [n for n in names if any(t in n for t in ("test_labels", "test_meta", "private", "parent_map")) or os.path.basename(n) == "manifest.csv"]
            if sp != "test_inputs.zip" and "manifest_public.csv" in names:
                z = zipfile.ZipFile(os.path.join(DS, "main20", sp)); mp = pd.read_csv(z.open("manifest_public.csv"))
                if (mp.split == "test").any(): bad.append("manifest_public.csv has test rows")
            if sp == "test_inputs.zip":
                bad += [n for n in names if not n.startswith(("test/video/", "test/audio/")) and n != "test_manifest_inputs.csv"]
            ok &= not bad
            print(f"{sp}: {len(names)}개 파일, 문제 {bad[:5]}")
        else:
            print(f"{sp}: {len(names)}개 파일 (비공개)")
    m = pd.read_csv(os.path.join(DS, "main20", "manifest.csv"))
    z = zipfile.ZipFile(os.path.join(DS, "main20", "test_inputs.zip")).namelist()
    nv = sum(n.endswith(".mp4") for n in z); print("test_inputs 영상 수", nv, "기대", int(((m.split == "test") & (m.dropped == 0)).sum()))
    ok &= nv == int(((m.split == "test") & (m.dropped == 0)).sum())
    print("ctrl5 공개 zip 없음:", not glob.glob(os.path.join(DS, "ctrl5", "*.zip")))
    print("V5", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["1", "2", "3", "4", "5"]):
        print(f"===== 확인 {k} =====")
        {"1": v1, "2": v2, "3": v3, "4": v4, "5": v5}[k]()
