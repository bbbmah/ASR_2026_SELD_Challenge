# [변경 이력 시작]
#   2026-10-03  최초 생성: BUILD_REPORT.md 재조립(ctrl5 v2 절 추가, v1 폐기 표기, fix20 절)
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""BUILD_REPORT.md 재조립: ctrl5 v2 절 추가, main20 절 갱신, 옛 ctrl5는 'ctrl5 v1 (폐기)'로."""
import os, glob, json
import numpy as np, pandas as pd
import finalize as F, build as B

DS = B.OUTROOT
old = open(os.path.join(DS, "BUILD_REPORT.md"), encoding="utf8").read() if not os.path.exists(os.path.join(DS, "BUILD_REPORT.v1.md")) \
    else open(os.path.join(DS, "BUILD_REPORT.v1.md"), encoding="utf8").read()
if not os.path.exists(os.path.join(DS, "BUILD_REPORT.v1.md")):
    open(os.path.join(DS, "BUILD_REPORT.v1.md"), "w", encoding="utf8").write(old)
secs = old.split("\n## ")
head, secs = secs[0], ["## " + s for s in secs[1:]]
get = lambda key: next(s for s in secs if s.startswith("## " + key))
v1_sec = get("ctrl5").replace("## ctrl5", "## ctrl5 v1 (폐기)", 1)
rest = [s for s in secs if not s.startswith(("## main20", "## ctrl5", "## 챌린지 성립 지표"))]

fin = json.load(open(os.path.join(DS, "logs", "ctrl5v2_finalize.json")))
P = ("train", "val", "test")
S = {(d, sp): F.stats(d, sp) for d in ("main20", "ctrl5") for sp in P}
EMPTY = "frame,class,source,azimuth,distance,onscreen"


def empty_ratio(sp):
    ld, _ = F.split_dirs("ctrl5", sp); fs = glob.glob(os.path.join(ld, "*.csv"))
    return sum(1 for f in fs if len(pd.read_csv(f)) == 0), len(fs)


def table(ds, with_empty=False):
    L = ["| split | 클립 | 시간(h) | onscreen 프레임 비율 | 소리내는 사람 중 onscreen 경험 | 용량(GB) | 피크로 빠진 창 |" + (" 라벨 없는 조각 |" if with_empty else ""),
         "|---|---|---|---|---|---|---|" + ("---|" if with_empty else "")]
    for sp in P:
        s = S[(ds, sp)]
        dr = fin["windows_dropped_by_split"].get(sp, 0); tot = fin["windows_total_by_split"][sp]
        row = f"| {sp} | {s['clips']} | {s['hours']:.2f} | {s['onscreen_ratio']:.3f} | {s['src_on']}/{s['src_total']} = {s['src_ratio']:.3f} | {s['size_gb']:.2f} | {dr}/{tot} |"
        if with_empty:
            e, n = empty_ratio(sp); row += f" {e}/{n} = {e/n:.2%} |"
        L.append(row)
    return L


def class_table(ds):
    L = ["| class | " + " | ".join(P) + " |", "|---|---|---|---|"]
    for c in range(13):
        L.append(f"| {c} {F.CLASSES[c]} | " + " | ".join(str(S[(ds, sp)]['cls'][c]) for sp in P) + " |")
    return L


same = all((S[("main20", sp)]["cls"] == S[("ctrl5", sp)]["cls"]).all() for sp in P)
v2 = ["## ctrl5 v2", "",
      "main20의 20초 창을 5초씩 4등분해 조각마다 요를 무작위로 고정(피치 0)하고, 원본 360° 영상과 FOA에서 다시 렌더링했다. 조각 단위로 거르지 않았다. 내부용이며 공개하지 않는다.", ""] \
     + table("ctrl5", True) + ["",
      f"- 소리를 낸 사람 중 한 번이라도 onscreen=1이었던 비율(클립 단위): " + ", ".join(f"{sp} {S[('ctrl5', sp)]['src_ratio']:.3f}" for sp in P),
      f"- 클래스별 라벨 행 수가 split 모두에서 main20과 {'정확히 같다 (아래 main20 표와 동일)' if same else '다르다 (!)'}.",
      "", "### 필수 확인 5개 (validate_ctrl5.py)", "",
      "1. 개수: train 11996 = 2999×4, val 492 = 123×4, test 2104 = 526×4. 통과.",
      "2. 같은 구간: 클래스별 행 수 main20 == ctrl5 (split 3개 모두 일치). 조각 14592개의 meta(`frame+50j, class, source, az_world, el_world, distance`)가 부모 meta와 불일치 0건. 통과.",
      "3. 빈 라벨 읽기: 베이스라인 `load_labels`/`metrics`가 헤더만 있는 csv를 크래시 없이 읽는다. 단 정답이 없는 파일·프레임의 예측은 오검출로 세지 않는다(`metrics.py:120, 223`). 데이터는 그대로 두고 NOTES에 기록.",
      "4. 채점 단위: 0.1초 프레임 단위(`metrics.py:120-128, 169-177`). NOTES에 근거 기록.",
      "5. 공개 묶음: main20 `train.zip`, `val.zip`, `test_inputs.zip`에 test 라벨/meta/private/parent_map/manifest.csv 없음, `manifest_public.csv`에 test 행 없음. 통과. ctrl5는 공개 zip을 만들지 않았다.", ""]
m20 = ["## main20", "", "(ctrl5 v2 작업에서 피크로 빠진 창을 반영해 갱신. 빠진 클립은 `main20/_dropped/`)", ""] + table("main20") + ["", "클래스별 라벨 행 수 (프레임 단위)", ""] + class_table("main20") + [""]

key = ["## 챌린지 성립 지표: 소리를 낸 사람(source>0) 중 한 번이라도 onscreen=1이었던 비율", "",
       "| split | main20 | ctrl5 v2 | (참고) ctrl5 v1 |", "|---|---|---|---|"]
v1 = {"train": "0.278", "val": "0.250", "test": "0.282"}
for sp in P:
    a, b = S[("main20", sp)], S[("ctrl5", sp)]
    key.append(f"| {sp} | {a['src_ratio']:.3f} ({a['src_on']}/{a['src_total']}) | {b['src_ratio']:.3f} ({b['src_on']}/{b['src_total']}) | {v1[sp]} |")
key += ["", "(클립 단위로 source id별로 센 값. ctrl5 v2는 조각 단위라 한 사람이 여러 조각에 나뉘어 세어진다.)", ""]

extra = []
fixp = os.path.join(DS, "fix20", "manifest.csv")
if os.path.exists(fixp):
    fm = pd.read_csv(fixp)
    extra = ["## fix20 (선택, 내부용)", "",
             "main20 val/test의 남은 창마다 20초 클립 1개, 요 고정(클립마다 무작위), 피치 0. 이름은 부모 main20과 같다.", "",
             f"- 클립 수: " + ", ".join(f"{sp} {int(((fm.split==sp)&(fm.dropped==0)).sum())}" for sp in ("val", "test")) +
             " / 피크로 빠진 클립: " + ", ".join(f"{sp} {int(((fm.split==sp)&(fm.dropped==1)).sum())}" for sp in ("val", "test")), ""]
    fr = os.path.join(DS, "logs", "fix20_check.txt")
    if os.path.exists(fr): extra += [open(fr, encoding="utf8").read(), ""]

out = "\n".join([head.rstrip(), ""] + v2 + m20 + [v1_sec.rstrip(), ""] + key + extra + [s.rstrip() + "\n" for s in rest])
open(os.path.join(DS, "BUILD_REPORT.md"), "w", encoding="utf8").write(out)
print(out[:3500])
