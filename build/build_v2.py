# [변경 이력 시작]
#   2026-10-03  최초 생성: ctrl5 v2(main20 창을 5초 4조각, 고정 카메라)와 fix20 생성, 피크 탈락 반영, 이름 매기기
#   2026-10-03  pandas 메서드와 겹치는 m.clip을 m["clip"]으로 수정, fix20 파일 이동 코드 단순화
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""ctrl5 v2 (main20 창을 5초씩 4등분, 고정 카메라) 와 fix20 생성. 렌더링 코드는 dsgen 그대로 사용.
python build_v2.py plan            # 조각 목록(plan_ctrl5v2.csv) 생성
python build_v2.py render ctrl5v2 [--workers N]
python build_v2.py finalize        # 피크 탈락 반영, 이름 붙이기, parent_map
python build_v2.py fix20 [--workers N]
"""
import sys, os, json, time, csv, shutil, argparse
import numpy as np, pandas as pd, soundfile as sf
from multiprocessing import Pool
import dsgen as G
import build as B

W = r"C:\asrwork"
DS = os.path.join(W, "dataset")
TMP = os.path.join(W, "v2_tmp")
SEED = B.SEED
SPLITS = ("train", "val", "test")


def parents():
    return pd.read_csv(os.path.join(DS, "main20", "manifest.csv"))


# ------------------------------------------------------------------ 공통 작업자
_PROBE = None


def render_job(job):
    """job: dict(root, ds, split, clip, recording, start, n, psi)"""
    global _PROBE
    try:
        if _PROBE is None:
            _PROBE = {p["name"]: p for p in B.load_valid()}
        p = _PROBE[job["recording"]]
        P = B.paths(job["root"], job["ds"], job["split"], job["clip"])
        side = os.path.join(job["root"], "_peak", job["ds"], job["clip"] + ".txt")
        if all(os.path.exists(x) for x in P.values()) and os.path.exists(side):
            return job["clip"], "skip", float(open(side).read())
        for x in list(P.values()) + [side]: os.makedirs(os.path.dirname(x), exist_ok=True)
        n, start = int(job["n"]), int(job["start"])
        tr = G.Traj("fixed", n / 10, float(job["psi"]))
        a = G.clip_audio(p["wav"], start, n, tr)
        peak = float(np.abs(a).max())
        lab = G.load_labels(p["csv"])
        pub, ext, trj = G.clip_labels(lab, start, n, tr)
        tmp = {k: v + ".tmp" for k, v in P.items()}
        sf.write(tmp["audio"], a.T, G.SR, format="WAV", subtype="PCM_16")
        pub.to_csv(tmp["lab"], index=False); ext.to_csv(tmp["ext"], index=False); trj.to_csv(tmp["traj"], index=False)
        G.write_video(p["mp4"], p["fps"], start, n, tr, P["video"] + ".tmp.mp4")
        for k in ("audio", "lab", "ext", "traj"): os.replace(tmp[k], P[k])
        open(side, "w").write(repr(peak))
        os.replace(P["video"] + ".tmp.mp4", P["video"])           # 영상이 마지막 = 완료 표시
        return job["clip"], "ok", peak
    except Exception as e:
        return job["clip"], f"fail: {type(e).__name__}: {e}", float("nan")


def run_jobs(jobs, workers, tag):
    fl = os.path.join(DS, "logs", "failures.csv"); os.makedirs(os.path.dirname(fl), exist_ok=True)
    t0 = time.time(); ok = skip = fail = 0; peaks = {}
    with Pool(workers) as pool:
        for i, (clip, st, pk) in enumerate(pool.imap_unordered(render_job, jobs, chunksize=1)):
            if st in ("ok", "skip"):
                peaks[clip] = pk; ok += st == "ok"; skip += st == "skip"
            else:
                fail += 1
                new = not os.path.exists(fl)
                with open(fl, "a", newline="", encoding="utf8") as f:
                    w = csv.writer(f)
                    if new: w.writerow(["dataset", "split", "clip", "error"])
                    w.writerow([tag, "", clip, st])
            if (i + 1) % 200 == 0 or i + 1 == len(jobs):
                print(f"{tag} {i+1}/{len(jobs)} ok={ok} skip={skip} fail={fail} {time.time()-t0:.0f}s", flush=True)
    print(f"DONE {tag}: ok={ok} skip={skip} fail={fail} wall={time.time()-t0:.1f}s", flush=True)
    return peaks


# ------------------------------------------------------------------ plan
def plan():
    m = parents(); rows = []
    for si, sp in enumerate(SPLITS):
        rng = np.random.default_rng([SEED, 3, B.SPLIT_ID[sp]])   # 'ctrl5v2' = 데이터셋 id 3
        for r in m[m.split == sp].itertuples():
            for j in range(4):
                rows.append(dict(tmp=f"{r.clip}_{j}", split=sp, parent_clip=r.clip, j=j, recording=r.recording, room=r.room,
                                 parent_start=r.start_frame, start_frame=r.start_frame + 50 * j, n_frames=50,
                                 psi=int(rng.integers(0, 360))))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(W, "plan_ctrl5v2.csv"), index=False)
    print(d.groupby("split").size().to_dict())


def render_ctrl5v2(workers):
    d = pd.read_csv(os.path.join(W, "plan_ctrl5v2.csv"))
    jobs = [dict(root=TMP, ds="ctrl5", split=r.split, clip=r.tmp, recording=r.recording, start=r.start_frame, n=50, psi=r.psi)
            for r in d.itertuples()]
    peaks = run_jobs(jobs, workers, "ctrl5v2")
    pd.Series(peaks, name="peak").rename_axis("tmp").reset_index().to_csv(os.path.join(W, "peaks_ctrl5v2.csv"), index=False)


# ------------------------------------------------------------------ finalize
def move_files(P_from, P_to):
    for k in P_from:
        if os.path.exists(P_from[k]):
            os.makedirs(os.path.dirname(P_to[k]), exist_ok=True)
            os.replace(P_from[k], P_to[k])


def finalize():
    d = pd.read_csv(os.path.join(W, "plan_ctrl5v2.csv"))
    pk = pd.read_csv(os.path.join(W, "peaks_ctrl5v2.csv")).set_index("tmp").peak
    d["peak"] = d.tmp.map(pk)
    assert d.peak.notna().all(), "렌더링이 안 끝난 조각이 있음"
    bad = d[d.peak > G.PEAK_LIMIT].groupby("parent_clip").size()
    dropped = set(bad.index)
    print("피크로 빠지는 창:", d[d.parent_clip.isin(dropped)].drop_duplicates("parent_clip").groupby("split").size().to_dict(), "총", len(dropped))
    # main20 manifest에 dropped 열
    mp = os.path.join(DS, "main20", "manifest.csv"); m = pd.read_csv(mp)
    if "dropped" not in m.columns:
        m["dropped"] = m["clip"].isin(dropped).astype(int)
        shutil.copy(mp, mp + ".bak_before_v2")
        m.to_csv(mp, index=False)
    # main20 클립 파일 이동
    for r in m[m.dropped == 1].itertuples():
        Pf = B.paths(DS, "main20", r.split, r.clip)
        Pt = B.paths(os.path.join(DS, "main20", "_dropped"), ".", r.split, r.clip)
        move_files(Pf, Pt)
    # ctrl5 최종 이름
    keep = d[~d.parent_clip.isin(dropped)].copy()
    out = []
    for sp in SPLITS:
        k = keep[keep.split == sp].reset_index(drop=True)          # 계획 순서 = 부모 순서, j 순서
        k["clip"] = [f"{sp}_{i:05d}" for i in range(len(k))]
        out.append(k)
        for r in k.itertuples():
            move_files(B.paths(TMP, "ctrl5", sp, r.tmp), B.paths(DS, "ctrl5", sp, r.clip))
    for r in d[d.parent_clip.isin(dropped)].itertuples():             # 빠진 조각은 지우지 않고 옮겨 둠
        move_files(B.paths(TMP, "ctrl5", r.split, r.tmp), B.paths(os.path.join(DS, "ctrl5", "_dropped"), ".", r.split, r.tmp))
    k = pd.concat(out)
    pm = k[["clip", "parent_clip", "j", "start_frame", "recording", "psi"]]
    os.makedirs(os.path.join(DS, "ctrl5", "private"), exist_ok=True)
    pm.to_csv(os.path.join(DS, "ctrl5", "private", "parent_map.csv"), index=False)
    man = k[["clip", "split", "recording", "room", "start_frame", "n_frames", "psi", "parent_clip", "j", "peak"]].copy()
    man.insert(5, "kind", "fixed"); man = man.rename(columns={"psi": "psi0"})
    man.to_csv(os.path.join(DS, "ctrl5", "manifest.csv"), index=False)
    shutil.copy(os.path.join(DS, "ctrl5_v1_old", "split_info.json"), os.path.join(DS, "ctrl5", "split_info.json"))
    st = dict(windows_dropped_by_split=d[d.parent_clip.isin(dropped)].drop_duplicates("parent_clip").groupby("split").size().to_dict(),
              windows_total_by_split=d.drop_duplicates("parent_clip").groupby("split").size().to_dict(),
              pieces_kept=keep.groupby("split").size().to_dict())
    json.dump(st, open(os.path.join(DS, "logs", "ctrl5v2_finalize.json"), "w"), indent=1)
    print(st)


# ------------------------------------------------------------------ fix20
def fix20(workers):
    m = parents(); m = m[(m.dropped == 0) & m.split.isin(["val", "test"])]
    jobs = []
    for sp in ("val", "test"):
        rng = np.random.default_rng([SEED, 4, B.SPLIT_ID[sp]])
        for r in m[m.split == sp].itertuples():
            jobs.append(dict(root=os.path.join(W, "fix20_tmp"), ds="fix20", split=sp, clip=r.clip, recording=r.recording,
                             start=r.start_frame, n=200, psi=int(rng.integers(0, 360))))
    pd.DataFrame(jobs).drop(columns=["root"]).to_csv(os.path.join(W, "plan_fix20.csv"), index=False)
    peaks = run_jobs(jobs, workers, "fix20")
    root = os.path.join(W, "fix20_tmp"); fixdir = os.path.join(DS, "fix20")
    dropped = []
    for j in jobs:
        ok = peaks.get(j["clip"], 9) <= G.PEAK_LIMIT
        if not ok: dropped.append(j["clip"])
        dst = B.paths(DS, "fix20", j["split"], j["clip"]) if ok else B.paths(os.path.join(fixdir, "_dropped"), ".", j["split"], j["clip"])
        move_files(B.paths(root, "fix20", j["split"], j["clip"]), dst)
    man = pd.DataFrame(jobs).drop(columns=["root", "ds"]); man["peak"] = man["clip"].map(peaks); man["dropped"] = (man.peak > G.PEAK_LIMIT).astype(int)
    man.to_csv(os.path.join(fixdir, "manifest.csv"), index=False)
    print("fix20 dropped:", dropped)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("arg", nargs="?"); ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "plan": plan()
    elif a.cmd == "render": render_ctrl5v2(a.workers)
    elif a.cmd == "finalize": finalize()
    elif a.cmd == "fix20": fix20(a.workers)
