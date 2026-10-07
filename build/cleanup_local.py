# [변경 이력 시작]
#   2026-10-07  최초 생성: 로컬 정리 A, B, C. 기본은 dry-run, --execute일 때만 삭제하고 삭제 전 드라이브 사본 크기 대조, 삭제 목록 저장
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""로컬(C:\asrwork) 정리 A, B, C. 기본은 dry-run(삭제하지 않음). --execute 일 때만 삭제한다.
삭제 전에 모든 대상 파일이 드라이브에 같은 크기로 있는지 다시 대조하고, 하나라도 다르면 그 단계를 중단한다.
삭제 목록은 cleanup_inventory.csv 에 저장한다."""
import os, sys, csv, shutil, time
L = r"C:\asrwork"; D = r"G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset"
EXEC = "--execute" in sys.argv
only = [a for a in sys.argv[1:] if a in ("A", "B", "C")] or ["A", "B", "C"]

def walk(root):
    st = [root]
    while st:
        p = st.pop()
        with os.scandir(p) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False): st.append(e.path)
                else: yield e.path, e.stat().st_size

def free_gb():
    return shutil.disk_usage("C:\\").free / 2**30

# 대상 정의 ------------------------------------------------------------
def targets(tier):
    ds = os.path.join(L, "dataset"); dirs, files = [], []
    if tier == "A":
        dirs += [os.path.join(ds, "main20", "_dropped", "old_zips"), os.path.join(ds, "ctrl5_v1_old")]
    if tier == "B":
        for d in ("main20", "ctrl5", "fix20"):
            for sp in ("train", "val", "test"):
                for sub in ("video", "audio"):
                    p = os.path.join(ds, d, sp, sub)
                    if os.path.isdir(p): dirs.append(p)
    if tier == "C":
        for z in ("train.zip", "val.zip", "test_inputs.zip"):
            files.append(os.path.join(ds, "main20", z))
        dirs.append(os.path.join(L, "internal"))
    return dirs, files

inv = open(os.path.join(L, "cleanup_inventory.csv"), "a", newline="", encoding="utf8"); w = csv.writer(inv)
if inv.tell() == 0: w.writerow(["tier", "path", "bytes"])
print("모드:", "삭제 실행" if EXEC else "dry-run(삭제 안 함)", "| C: 여유 %.1f GB" % free_gb(), flush=True)
for tier in only:
    dirs, files = targets(tier)
    allf = []
    for d in dirs:
        assert d.startswith(L + "\\") and os.path.isdir(d), d
        allf += list(walk(d))
    for f in files:
        assert f.startswith(L + "\\") and os.path.isfile(f), f
        allf.append((f, os.path.getsize(f)))
    total = sum(s for _, s in allf); t0 = time.time(); bad = []
    for p, s in allf:                                     # 드라이브 사본 확인
        dp = os.path.join(D, os.path.relpath(p, L)) if not p.startswith(os.path.join(L, "internal")) else os.path.join(D, os.path.relpath(p, L))
        try: ok = os.stat(dp).st_size == s
        except OSError: ok = False
        if not ok: bad.append(p)
    print(f"[{tier}] 대상 파일 {len(allf)}개, {total/2**30:.2f} GB | 드라이브 사본 확인 {time.time()-t0:.0f}초, 없거나 크기 다름: {len(bad)}", flush=True)
    if bad:
        print("  중단. 예:", bad[:5]); break
    if EXEC:
        for p, s in allf: w.writerow([tier, p, s])
        inv.flush()
        for f in files: os.remove(f)
        for d in dirs: shutil.rmtree(d)
        print(f"[{tier}] 삭제 완료 | C: 여유 %.1f GB" % free_gb(), flush=True)
inv.close()
