# [변경 이력 시작]
#   2026-10-06  최초 생성: 로컬과 드라이브 사본의 파일 목록, 크기, zip 목록과 CRC를 읽기 전용으로 대조
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""로컬(C:\asrwork)과 드라이브 사본이 같은지 읽기 전용으로 점검한다. 삭제하지 않는다.
1) dataset/, internal/ 의 모든 파일: 드라이브에 있는지, 크기가 같은지
2) 모든 zip: 로컬과 드라이브의 중앙 디렉터리(이름, 크기, CRC)가 같은지 + 드라이브 zip의 데이터 CRC 검사(testzip)
결과: C:\asrwork\verify_sync.json"""
import os, sys, json, time, zipfile
L = r"C:\asrwork"; D = r"G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset"
PAIRS = [(os.path.join(L, "dataset"), os.path.join(D, "dataset")), (os.path.join(L, "internal"), os.path.join(D, "internal"))]
out = dict(files={}, zips=[])

def walk(root):
    st = [root]
    while st:
        p = st.pop()
        with os.scandir(p) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False): st.append(e.path)
                else: yield e.path, e.stat().st_size

t0 = time.time()
zips = []
for lroot, droot in PAIRS:
    n = bad = missing = 0; total = 0; samples = []
    for path, size in walk(lroot):
        rel = os.path.relpath(path, lroot); dp = os.path.join(droot, rel); n += 1; total += size
        if path.lower().endswith(".zip"): zips.append((path, dp))
        try: ds = os.stat(dp).st_size
        except OSError: ds = None
        if ds is None: missing += 1; samples.append(("없음", rel))
        elif ds != size: bad += 1; samples.append(("크기다름", rel, size, ds))
    out["files"][os.path.basename(lroot)] = dict(local_files=n, local_GB=round(total / 1e9, 2), missing_on_drive=missing, size_mismatch=bad, samples=samples[:15])
    print(os.path.basename(lroot), out["files"][os.path.basename(lroot)], f"{time.time()-t0:.0f}s", flush=True)
for lp, dp in zips:
    r = dict(zip=os.path.relpath(lp, L), GB=round(os.path.getsize(lp) / 1e9, 2))
    try:
        zl, zd = zipfile.ZipFile(lp), zipfile.ZipFile(dp)
        il = [(i.filename, i.file_size, i.CRC) for i in zl.infolist()]; idr = [(i.filename, i.file_size, i.CRC) for i in zd.infolist()]
        r["entries"] = len(il); r["central_dir_equal"] = il == idr
        t1 = time.time(); bad = zd.testzip(); r["drive_testzip_bad_member"] = bad; r["testzip_sec"] = round(time.time() - t1)
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
    out["zips"].append(r); print(r, flush=True)
    json.dump(out, open(r"C:\asrwork\verify_sync.json", "w"), indent=1, ensure_ascii=False)
json.dump(out, open(r"C:\asrwork\verify_sync.json", "w"), indent=1, ensure_ascii=False)
print("ALLDONE", f"{time.time()-t0:.0f}s")
