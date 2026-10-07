# [변경 이력 시작]
#   2026-10-02  최초 생성: 통계, 챌린지 성립 지표, BUILD_REPORT 작성, 공개 zip 포장
#   2026-10-03  공개 묶음 만들 때 피크로 빠진(dropped) main20 클립을 제외하도록 수정
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""통계 -> BUILD_REPORT.md, README, zip 포장."""
import os, json, zipfile, glob
import numpy as np, pandas as pd
import soundfile as sf
import build as B

ROOT = B.OUTROOT
CLASSES = ["Female speech", "Male speech", "Clapping", "Telephone", "Laughter", "Domestic sounds", "Walk/footsteps",
           "Door open/close", "Music", "Musical instrument", "Water tap", "Bell", "Knock"]


def split_dirs(ds, sp):
    b = os.path.join(ROOT, ds)
    return (os.path.join(b, "private", "test_labels") if sp == "test" else os.path.join(b, sp, "labels"),
            os.path.join(b, "private", "test_meta") if sp == "test" else os.path.join(b, sp, "meta"))


def dir_gb(path):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs) / 1e9


def stats(ds, sp):
    ld, md = split_dirs(ds, sp)
    files = sorted(glob.glob(os.path.join(ld, "*.csv")))
    n = len(files)
    if n == 0: return None
    frames_per = B.CFG[ds]["n_frames"]
    cls = np.zeros(13, int); on = tot = 0; srcs = srcs_on = 0
    for f in files:
        d = pd.read_csv(f)
        cls += np.bincount(d["class"], minlength=13)[:13]
        on += d.onscreen.sum(); tot += len(d)
        s = d[d.source > 0]
        srcs += s.source.nunique(); srcs_on += s[s.onscreen == 1].source.nunique()
    return dict(clips=n, hours=n * frames_per / 36000, cls=cls, onscreen_ratio=on / tot, rows=tot,
                src_total=srcs, src_on=srcs_on, src_ratio=srcs_on / max(1, srcs),
                size_gb=dir_gb(os.path.join(ROOT, ds, sp)) + (dir_gb(os.path.join(ROOT, ds, "private")) if sp == "test" else 0))


def make_zips(ds):
    b = os.path.join(ROOT, ds)
    def pack(zname, folders):
        with zipfile.ZipFile(os.path.join(b, zname), "w", zipfile.ZIP_STORED) as z:
            for fol in folders:
                for r, _, fs in os.walk(os.path.join(b, fol)):
                    for f in fs:
                        p = os.path.join(r, f); z.write(p, os.path.relpath(p, b))
            for f in ("manifest_public.csv", "split_info.json"):
                if os.path.exists(os.path.join(b, f)): z.write(os.path.join(b, f), f)
    m = pd.read_csv(os.path.join(b, "manifest.csv"))
    if 'dropped' in m.columns: m = m[m.dropped == 0]
    # 공개 manifest: test 행에는 카메라 궤적 값이 없어야 하므로 clip/split/recording/start만 남긴다(녹음 이름은 방 정보 노출을 막기 위해 제외)
    pub = m[m.split != "test"]
    pub.to_csv(os.path.join(b, "manifest_public.csv"), index=False)
    test_in = m[m.split == "test"][["clip", "split", "n_frames"]]
    test_in.to_csv(os.path.join(b, "test_manifest_inputs.csv"), index=False)
    pack("train.zip", ["train"]); pack("val.zip", ["val"])
    with zipfile.ZipFile(os.path.join(b, "test_inputs.zip"), "w", zipfile.ZIP_STORED) as z:
        for sub in ("video", "audio"):
            for r, _, fs in os.walk(os.path.join(b, "test", sub)):
                for f in fs:
                    p = os.path.join(r, f); z.write(p, os.path.relpath(p, b))
        z.write(os.path.join(b, "test_manifest_inputs.csv"), "test_manifest_inputs.csv")
    with zipfile.ZipFile(os.path.join(b, "private_test.zip"), "w", zipfile.ZIP_STORED) as z:
        for r, _, fs in os.walk(os.path.join(b, "private")):
            for f in fs:
                p = os.path.join(r, f); z.write(p, os.path.relpath(p, b))
        z.write(os.path.join(b, "manifest.csv"), "manifest.csv")      # 비공개: 궤적·녹음 정보 포함


def report():
    L = ["# BUILD_REPORT", ""]
    allst = {}
    for ds in B.CFG:
        pl = json.load(open(os.path.join(ROOT, "logs", f"plan_{ds}.json")))
        L += [f"## {ds}", "", "| split | 클립 | 시간(h) | onscreen 프레임 비율 | 소리내는 사람 중 onscreen 경험 | 용량(GB) | 피크 탈락 | 라벨<30프레임 탈락 |",
              "|---|---|---|---|---|---|---|---|"]
        for sp in ("train", "val", "test"):
            s = stats(ds, sp); allst[(ds, sp)] = s
            if s is None: continue
            L.append(f"| {sp} | {s['clips']} | {s['hours']:.2f} | {s['onscreen_ratio']:.3f} | {s['src_on']}/{s['src_total']} = {s['src_ratio']:.3f} | {s['size_gb']:.2f} | {pl[sp]['clip_drop_ratio']:.3%} | {pl[sp]['few_label_ratio']:.3%} |")
        L += ["", "클래스별 라벨 행 수 (프레임 단위)", "", "| class | " + " | ".join(sp for sp in ("train", "val", "test")) + " |", "|---|---|---|---|"]
        for c in range(13):
            L.append(f"| {c} {CLASSES[c]} | " + " | ".join(str(allst[(ds, sp)]['cls'][c]) if allst[(ds, sp)] else "-" for sp in ("train", "val", "test")) + " |")
        L.append("")
    L += ["## 챌린지 성립 지표: 소리를 낸 사람(source>0) 중 한 번이라도 onscreen=1이었던 비율", "",
          "| split | main20 | ctrl5 |", "|---|---|---|"]
    for sp in ("train", "val", "test"):
        a, b = allst[("main20", sp)], allst[("ctrl5", sp)]
        L.append(f"| {sp} | {a['src_ratio']:.3f} ({a['src_on']}/{a['src_total']}) | {b['src_ratio']:.3f} ({b['src_on']}/{b['src_total']}) |")
    L += ["", "(클립 안에서 source id별로 센 값. 한 녹음이 여러 클립에 들어가면 클립마다 따로 센다.)", ""]
    L += open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_tail.md"), encoding="utf8").read().splitlines()
    open(os.path.join(ROOT, "BUILD_REPORT.md"), "w", encoding="utf8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    import sys
    if "zip" in sys.argv:
        for ds in B.CFG: make_zips(ds)
    if "report" in sys.argv:
        report()
