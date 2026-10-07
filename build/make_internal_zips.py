# [변경 이력 시작]
#   2026-10-03  최초 생성: ctrl5, fix20 내부 전송용 zip(internal_*) 만들기
#   2026-10-07  설명 문자열의 윈도우 경로를 슬래시로 바꿔 파이썬 SyntaxWarning('\i') 제거(동작 변경 없음)
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""ctrl5, fix20 내부 전송용 zip 만들기 (비공개). 공개 zip과 같은 폴더 구조(데이터셋 폴더 기준 상대 경로)로 묶는다.
internal_{dataset}_{train|val|test}.zip, internal_{dataset}_private.zip  ->  C:/asrwork/internal"""
import os, zipfile, time
DS = r"C:\asrwork\dataset"; OUT = r"C:\asrwork\internal"
os.makedirs(OUT, exist_ok=True)

def pack(zname, base, folders, files=()):
    path = os.path.join(OUT, zname)
    if os.path.exists(path):
        print("있음, 건너뜀", zname); return
    t0 = time.time(); n = 0
    with zipfile.ZipFile(path + ".tmp", "w", zipfile.ZIP_STORED) as z:
        for fol in folders:
            for r, _, fs in os.walk(os.path.join(base, fol)):
                for f in sorted(fs):
                    p = os.path.join(r, f); z.write(p, os.path.relpath(p, base)); n += 1
        for f in files:
            z.write(os.path.join(base, f), f); n += 1
    os.replace(path + ".tmp", path); print(f"{zname}: {n}개 파일, {os.path.getsize(path)/1e9:.2f}GB, {time.time()-t0:.0f}초", flush=True)

for ds, splits in (("ctrl5", ("train", "val", "test")), ("fix20", ("val", "test"))):
    base = os.path.join(DS, ds)
    for sp in splits:
        folders = ([f"{sp}/video", f"{sp}/audio"] if sp == "test" else [f"{sp}/video", f"{sp}/audio", f"{sp}/labels", f"{sp}/meta"])
        pack(f"internal_{ds}_{sp}.zip", base, folders)
    extra = [f for f in ("manifest.csv", "split_info.json") if os.path.exists(os.path.join(base, f))]
    pr = os.path.join(base, "private")
    pack(f"internal_{ds}_private.zip", base, ["private"], extra)
print("끝")
