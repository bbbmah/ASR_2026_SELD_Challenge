# [변경 이력 시작]
#   2026-10-02  최초 생성: 클립 계획(plan)과 병렬 생성(gen), main20과 ctrl5(v1). 이어 하기, 실패 로그 포함
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""데이터셋 계획(plan) 및 생성(gen).
python build.py plan
python build.py gen <main20|ctrl5> <train|val|test> [--workers N] [--limit N] [--out DIR]
"""
import sys, os, json, time, argparse, csv
import numpy as np, pandas as pd, soundfile as sf
from multiprocessing import Pool
import dsgen as G

WORK = r"C:\asrwork"
OUTROOT = os.path.join(WORK, "dataset")
SEED = 0
CFG = {"main20": dict(n_frames=200, kind="sweep", n_train=3000, sid=1),
       "ctrl5": dict(n_frames=50, kind="fixed", n_train=12000, sid=2)}
SPLIT_ID = {"train": 0, "val": 1, "test": 2}


def load_valid():
    return json.load(open(os.path.join(WORK, "probe_valid.json")))


def plan(ds):
    cfg = CFG[ds]; n = cfg["n_frames"]; clip_s = n / 10
    valid = load_valid()
    split_of, info = G.make_split(valid)
    labs = {p["name"]: G.load_labels(p["csv"]) for p in valid}
    labn = {k: set(v.frame.unique()) for k, v in labs.items()}
    recs = {s: [p for p in valid if split_of[p["name"]] == s and p["len_frames"] >= n] for s in SPLIT_ID}
    rows, stats = [], {}

    def try_clip(p, start, rng):
        tr = G.Traj.sample(cfg["kind"], clip_s, rng)           # 항상 같은 수의 난수 소비
        nl = sum(1 for f in range(start, start + n) if f in labn[p["name"]])
        if nl < G.MIN_LABELED:
            return None, "few_labels"
        a = G.clip_audio(p["wav"], start, n, tr)
        if np.abs(a).max() > G.PEAK_LIMIT:
            return None, "clip"
        return tr, None

    for split in ("val", "test", "train"):
        rng = np.random.default_rng([SEED, cfg["sid"], SPLIT_ID[split]])
        rej = dict(few_labels=0, clip=0); out = []
        if split == "train":
            w = np.array([p["len_frames"] for p in recs[split]], float); w /= w.sum()
            while len(out) < cfg["n_train"]:
                p = recs[split][rng.choice(len(w), p=w)]
                start = int(rng.integers(0, p["len_frames"] - n + 1))
                tr, why = try_clip(p, start, rng)
                if why: rej[why] += 1
                else: out.append((p, start, tr))
        else:
            for p in sorted(recs[split], key=lambda q: q["name"]):
                for start in range(0, p["len_frames"] - n + 1, n):
                    tr, why = try_clip(p, start, rng)
                    if why: rej[why] += 1
                    else: out.append((p, start, tr))
        tried = len(out) + sum(rej.values())
        stats[split] = dict(clips=len(out), tried=tried, rejected=rej,
                            clip_drop_ratio=rej["clip"] / max(1, tried), few_label_ratio=rej["few_labels"] / max(1, tried))
        print(ds, split, stats[split], flush=True)
        for i, (p, start, tr) in enumerate(out):
            rows.append(dict(clip=f"{split}_{i:05d}", split=split, recording=p["name"], room=p["room"],
                             start_frame=start, n_frames=n, **tr.params()))
    d = os.path.join(OUTROOT, ds); os.makedirs(d, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(d, "manifest.csv"), index=False)
    json.dump(info, open(os.path.join(d, "split_info.json"), "w"), indent=1, ensure_ascii=False)
    os.makedirs(os.path.join(OUTROOT, "logs"), exist_ok=True)
    json.dump(stats, open(os.path.join(OUTROOT, "logs", f"plan_{ds}.json"), "w"), indent=1)


def paths(root, ds, split, clip):
    b = os.path.join(root, ds)
    if split == "test":
        lab, meta = os.path.join(b, "private", "test_labels"), os.path.join(b, "private", "test_meta")
    else:
        lab, meta = os.path.join(b, split, "labels"), os.path.join(b, split, "meta")
    return dict(video=os.path.join(b, split, "video", clip + ".mp4"), audio=os.path.join(b, split, "audio", clip + ".wav"),
                lab=os.path.join(lab, clip + ".csv"), ext=os.path.join(meta, clip + "_ext.csv"),
                traj=os.path.join(meta, clip + "_traj.csv"))


_PROBE = None


def build_clip(job):
    global _PROBE
    row, root = job
    ds = "main20" if row["n_frames"] == 200 else "ctrl5"
    try:
        if _PROBE is None:
            _PROBE = {p["name"]: p for p in load_valid()}
        p = _PROBE[row["recording"]]
        P = paths(root, ds, row["split"], row["clip"])
        if all(os.path.exists(x) for x in P.values()):
            return row["clip"], "skip", 0.0
        for x in P.values(): os.makedirs(os.path.dirname(x), exist_ok=True)
        t0 = time.time(); n = int(row["n_frames"]); start = int(row["start_frame"])
        tr = G.Traj.from_params(n / 10, {k: row[k] for k in ("kind", "psi0", "s", "omega", "P", "phi0")})
        a = G.clip_audio(p["wav"], start, n, tr)
        if np.abs(a).max() > G.PEAK_LIMIT:
            raise RuntimeError("peak>0.9999 at build")
        lab = G.load_labels(p["csv"])
        pub, ext, trj = G.clip_labels(lab, start, n, tr)
        tmp = {k: v + ".tmp" for k, v in P.items()}
        sf.write(tmp["audio"], a.T, G.SR, format="WAV", subtype="PCM_16")
        pub.to_csv(tmp["lab"], index=False); ext.to_csv(tmp["ext"], index=False); trj.to_csv(tmp["traj"], index=False)
        G.write_video(p["mp4"], p["fps"], start, n, tr, P["video"] + ".tmp.mp4")
        for k in ("audio", "lab", "ext", "traj"): os.replace(tmp[k], P[k])
        os.replace(P["video"] + ".tmp.mp4", P["video"])           # 영상이 마지막 = 완료 표시
        return row["clip"], "ok", time.time() - t0
    except Exception as e:
        return row["clip"], f"fail: {type(e).__name__}: {e}", 0.0


def gen(ds, split, workers, limit, root):
    m = pd.read_csv(os.path.join(OUTROOT, ds, "manifest.csv"))
    m = m[m.split == split]
    if limit: m = m.head(limit)
    jobs = [(r, root) for r in m.to_dict("records")]
    os.makedirs(os.path.join(root, "logs"), exist_ok=True)
    fl = os.path.join(root, "logs", "failures.csv")
    t0 = time.time(); ok = skip = fail = 0; tt = []
    with Pool(workers) as pool:
        for i, (clip, st, dt) in enumerate(pool.imap_unordered(build_clip, jobs, chunksize=1)):
            if st == "ok": ok += 1; tt.append(dt)
            elif st == "skip": skip += 1
            else:
                fail += 1
                new = not os.path.exists(fl)
                with open(fl, "a", newline="", encoding="utf8") as f:
                    w = csv.writer(f)
                    if new: w.writerow(["dataset", "split", "clip", "error"])
                    w.writerow([ds, split, clip, st])
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"{ds}/{split} {i+1}/{len(jobs)} ok={ok} skip={skip} fail={fail} {time.time()-t0:.0f}s", flush=True)
    wall = time.time() - t0
    print(f"DONE {ds}/{split}: ok={ok} skip={skip} fail={fail} wall={wall:.1f}s "
          f"per-clip wall={wall/max(1,ok):.2f}s mean cpu/clip={np.mean(tt) if tt else 0:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("ds", nargs="?"); ap.add_argument("split", nargs="?")
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=OUTROOT)
    a = ap.parse_args()
    if a.cmd == "plan":
        for ds in ([a.ds] if a.ds else CFG): plan(ds)
    else:
        gen(a.ds, a.split, a.workers, a.limit, a.out)
