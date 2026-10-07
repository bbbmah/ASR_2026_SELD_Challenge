# [변경 이력 시작]
#   2026-10-07  최초 생성: 추가 데이터 생성 모듈(균일 격자 창 계획, 카메라 가족 F1~F5, 좌우 거울, 창 하나에서 main20 클립과 ctrl5 조각 생성, GPU 렌더러 옵션, 묶음 생성/zip/업로드 검증/이어 하기, 자체 점검)
#   2026-10-07  영상 임시 파일 이름을 .tmp.mp4로(OpenCV 컨테이너), JSON 출력 정리, 작은 목표에서 격자 간격 발산 방지, 매니페스트 병합 함수 추가
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
gen_v2.py -- DATA_PLAN_v2.md 의 추가 데이터 생성 (시간 창, 카메라 가족, 좌우 거울). 코랩 노트북(colab_generate.ipynb)이 이 모듈을 부른다.

기존 코드(dsgen.py)의 렌더링, 오디오 변환, 라벨 변환은 그대로 쓰고 아래만 새로 만든다.
 - 카메라 궤적 가족 F1~F5 (FamTraj), 거울(월드를 y -> -y 로 반사: FOA 의 Y 부호, 영상 좌우 뒤집기, 방위각 부호)
 - 균일 격자 시간 창 계획(make_plan)
 - 창 하나를 main20 클립 1개 + ctrl5 조각 4개로 한 번에 생성(gen_window). 같은 원본 프레임을 디코딩 한 번으로 공유한다.
 - 선택적 GPU 렌더러(GPURenderer): cv2.remap 과 같은 양선형 샘플링. 코랩에서 CPU 렌더러와 비교 검증 후 쓴다.
 - 묶음(shard) 단위 생성 -> zip -> 드라이브 복사 -> 검증 -> 로컬 삭제(run_generation)
"""
import os, sys, json, math, time, csv, glob, shutil, zipfile, traceback
import numpy as np
import pandas as pd
import cv2
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsgen as G

SEED = 0
N_MAIN, N_PIECE = 200, 50
RARE = (2, 3, 7, 10, 11, 12)        # 박수, 전화, 문, 물, 종, 노크
DEFAULT_MIX = {"F1": 0.50, "F2": 0.10, "F3": 0.10, "F4": 0.15, "F5": 0.15}
MAX_RATE = 45.0                      # 요 각속도 상한 (도/초)


# ====================================================================== 궤적
class FamTraj(G.Traj):
    """카메라 궤적 가족. psi/phi 는 도 단위, t 는 클립 시작 기준 초. dsgen 의 렌더링/오디오/라벨 코드가 그대로 쓴다."""
    def __init__(self, p, clip_s=20.0):
        super().__init__("fam", clip_s, float(p.get("psi0", 0.0)))
        self.p = dict(p)

    def psi(self, t):
        t = np.asarray(t, dtype=np.float64); p = self.p; fam = p["family"]
        if fam in ("F1", "F2", "F3"):
            return p["psi0"] + p["s"] * p["omega"] * t
        if fam == "F4":                                  # 정지 - 부드러운 회전(코사인 이징) 반복
            out = np.full_like(t, p["psi0"])
            for t0, du, d in zip(p["tk"], p["dur"], p["delta"]):
                x = np.clip((t - t0) / du, 0.0, 1.0)
                out = out + d * 0.5 * (1.0 - np.cos(np.pi * x))
            return out
        if fam == "F5":                                  # 좌우 왕복
            return p["psi0"] + p["A"] * np.sin(2 * np.pi * t / p["P"] + p["ph"])
        raise ValueError(fam)

    def phi(self, t):
        t = np.asarray(t, dtype=np.float64); p = self.p
        if p["pitch"] == "sin":
            return -12.5 + 12.5 * np.sin(2 * np.pi * t / p["Pp"] + p["phi0"])
        return np.full_like(t, float(p["pitch_c"]))

    def params(self):
        return dict(self.p)


class NegTraj:
    """요를 반대로 한 궤적 (거울 동치 검증용)."""
    def __init__(self, base): self.base = base
    def psi(self, t): return -self.base.psi(t)
    def phi(self, t): return self.base.phi(t)


def sample_traj(fam, clip_s, rng):
    p = {"family": fam}
    if fam in ("F1", "F2", "F3"):
        mult = {"F1": 1.0, "F2": 0.5, "F3": 2.0}[fam]
        p.update(psi0=float(rng.uniform(0, 360)), s=int(rng.choice([-1, 1])),
                 omega=float(360.0 / clip_s * mult * rng.uniform(0.9, 1.1)))
    elif fam == "F4":
        t = float(rng.uniform(1, 4)); tk, dur, delta = [], [], []
        while t < clip_s - 0.5:
            d = float(rng.choice([-1, 1]) * rng.uniform(30, 120))
            vp = float(rng.uniform(15, 40))                # 회전 중 최대 각속도
            du = abs(d) * (math.pi / 2) / vp
            tk.append(round(t, 4)); dur.append(round(du, 4)); delta.append(round(d, 3))
            t += du + float(rng.uniform(2, 6))
        p.update(psi0=float(rng.uniform(0, 360)), tk=tk, dur=dur, delta=delta)
    elif fam == "F5":
        A = float(rng.uniform(30, 90)); Pmin = max(8.0, A * 2 * math.pi / 40.0)
        p.update(psi0=float(rng.uniform(0, 360)), A=A, P=float(rng.uniform(Pmin, 20.0)), ph=float(rng.uniform(0, 2 * math.pi)))
    else:
        raise ValueError(fam)
    if fam == "F1":
        mode = "sin"
    else:
        u = rng.random(); mode = "sin" if u < 0.6 else ("c0" if u < 0.8 else "c12")
    if mode == "sin":
        p.update(pitch="sin", Pp=float(rng.uniform(10, 20)), phi0=float(rng.uniform(0, 2 * math.pi)))
    else:
        p.update(pitch="const", pitch_c=0.0 if mode == "c0" else -12.0)
    return p


def max_rate(traj, clip_s=20.0):
    t = np.arange(0, clip_s, 0.01)
    return float(np.abs(np.diff(traj.psi(t))).max() / 0.01)


# ====================================================================== 녹음 목록과 분할
def load_valid():
    """영상, 오디오, 라벨이 모두 있고 길이가 어긋나지 않는 녹음 (validate.py 검증 1과 같은 규칙)."""
    out = []
    for r in G.list_recordings():
        p = G.probe(r)
        if not (p["has_wav"] and p["has_mp4"]):
            continue
        if abs(p["audio_s"] - p["video_s"]) >= 5 or p["label_end_s"] - min(p["audio_s"], p["video_s"]) >= 5:
            continue
        if p["sr"] != G.SR or p["channels"] != 4 or (p["video_w"], p["video_h"]) != (1920, 960):
            continue
        out.append(p)
    return out


# ====================================================================== 창 계획
def _flags(p):
    lab = G.load_labels(p["csv"]); L = p["len_frames"]
    lf = np.zeros(L + 1, dtype=np.int32); rf = np.zeros(L + 1, dtype=np.int32)
    fr = lab.frame.values; ok = fr < L
    lf[np.unique(fr[ok])] = 1
    rare = lab[lab["class"].isin(RARE).values & ok]
    rf[np.unique(rare.frame.values)] = 1
    return np.concatenate([[0], np.cumsum(lf[:L])]), np.concatenate([[0], np.cumsum(rf[:L])])


def make_plan(valid, split_of, stage, n_target, mix=None, mirror_frac=0.5, rare_bonus=True, seed=SEED, jitter=5):
    """train 방 녹음에서 균일 격자 창을 뽑고(약 n_target 개), 창마다 카메라 가족, 거울, ctrl5 조각의 고정 요를 정한다."""
    mix = dict(mix or DEFAULT_MIX); fams = list(mix); pm = np.array([mix[f] for f in fams], float); pm /= pm.sum()
    recs = [p for p in valid if split_of[p["name"]] == "train" and p["len_frames"] >= N_MAIN]
    flags = {p["name"]: _flags(p) for p in recs}
    span_total = sum(p["len_frames"] - N_MAIN + 1 for p in recs)

    def lattice(delta, it):
        rng = np.random.default_rng([seed, 100 + stage, it]); rows = []
        for p in recs:
            span = p["len_frames"] - N_MAIN; lp, rp = flags[p["name"]]
            st = np.arange(rng.uniform(0, delta), span + 1e-9, delta)
            st = np.clip(np.round(st + rng.integers(-jitter, jitter + 1, len(st))), 0, span).astype(int)
            if st.size == 0: st = np.array([int(rng.integers(0, span + 1))])      # 간격이 녹음보다 길면 무작위 한 곳
            elif span - st.max() > delta / 2: st = np.append(st, span)
            for s in np.unique(st):
                if lp[s + N_MAIN] - lp[s] >= G.MIN_LABELED:
                    rows.append((p["name"], int(s), int(rp[s + N_MAIN] - rp[s])))
        return rows

    delta = span_total / max(1.0, n_target / (1.15 if rare_bonus else 1.0))
    for it in range(8):                                    # 목표 개수에 맞게 격자 간격을 조정
        base = lattice(delta, it)
        bonus = [r for r in base if r[2] >= 3] if rare_bonus else []
        total = len(base) + len(bonus)
        if abs(total - n_target) / n_target < 0.03: break
        delta = min(delta * total / n_target, span_total / 1.0)            # 격자 간격 조정 (작은 목표에서 발산하지 않게 상한)
    rng = np.random.default_rng([seed, 200 + stage])
    rows = [(*r, 0) for r in base] + [(*r, 1) for r in bonus]
    order = rng.permutation(len(rows)); room = {p["name"]: p["room"] for p in recs}
    out = []
    for w, i in enumerate(order):
        name, s, rc, bn = rows[i]
        fam = str(rng.choice(fams, p=pm))
        tp = sample_traj(fam, 20.0, rng)
        assert max_rate(FamTraj(tp)) <= MAX_RATE, (fam, max_rate(FamTraj(tp)))
        out.append(dict(w=w, clip=f"train_s{stage}_{w:05d}", recording=name, room=room[name], start_frame=s,
                        n_frames=N_MAIN, stage=stage, family=fam, mirror=int(rng.random() < mirror_frac), bonus=bn,
                        rare_frames=rc, traj=json.dumps(tp), piece_psi=json.dumps([int(x) for x in rng.integers(0, 360, 4)])))
    return pd.DataFrame(out), dict(delta_frames=delta, span_total_frames=span_total, n_base=len(base), n_bonus=len(bonus))


def plan_summary(plan, valid, split_of):
    T = sum(p["len_frames"] for p in valid if split_of[p["name"]] == "train") / 10.0
    n = len(plan)
    est_gb = (n * 3.56 + n * 4 * 0.635) / 1024
    return dict(windows=int(n), 영상시간_main20_h=round(n * 20 / 3600, 1), 시간커버_배수=round(n * 20 / T, 1),
                family={str(k): int(v) for k, v in plan.family.value_counts().items()}, mirror_비율=round(float(plan.mirror.mean()), 3),
                희귀보너스=int(plan.bonus.sum()), main20_클립=int(n), ctrl5_조각=int(n * 4), 예상용량_GB=round(est_gb, 1),
                방별창수={str(k): int(v) for k, v in plan.room.value_counts().sort_index().items()})


# ====================================================================== 렌더러
class CPURenderer:
    name = "cpu"
    def __init__(self): self.r = G.Renderer(); self.f = None
    def set_frame(self, frame): self.f = frame
    def render(self, psi, phi): return self.r.render(self.f, psi, phi)


class GPURenderer:
    """cv2.remap(양선형, 가로 wrap)과 같은 샘플링을 torch.grid_sample 로 한다. 장치가 cpu 여도 동작한다(검증용)."""
    name = "gpu"
    def __init__(self, device="cuda", in_hw=(960, 1920)):
        import torch
        self.torch = torch; self.dev = device; self.ih, self.iw = in_hw
        self.rays = torch.tensor(G.Renderer(in_hw).rays, dtype=torch.float32, device=device)
        self.img = None

    def set_frame(self, frame):
        t = self.torch
        x = t.from_numpy(frame).to(self.dev).permute(2, 0, 1).float()[None]          # (1,3,H,W), BGR 그대로
        self.img = t.cat([x[..., -2:], x, x[..., :2]], dim=-1)                          # 가로 wrap 을 위해 양쪽 2열 복사

    def render(self, psi, phi):
        t = self.torch
        fwd, left, up = G.cam_axes(np.float64(psi), np.float64(phi))
        M = t.tensor(np.stack([fwd, left, up], 1), dtype=t.float32, device=self.dev)
        d = self.rays @ M.T
        az = t.atan2(d[:, 1], d[:, 0]); el = t.atan2(d[:, 2], t.hypot(d[:, 0], d[:, 1]))
        mx = (0.5 - az / (2 * math.pi)) * self.iw - 0.5; my = (0.5 - el / math.pi) * self.ih - 0.5
        mx = t.remainder(mx, self.iw) + 2.0
        gx = (mx + 0.5) / (self.iw + 4) * 2 - 1; gy = (my + 0.5) / self.ih * 2 - 1
        grid = t.stack([gx, gy], -1).reshape(1, G.H, G.W, 2)
        out = t.nn.functional.grid_sample(self.img, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return out[0].permute(1, 2, 0).round().clamp(0, 255).to(t.uint8).cpu().numpy()


def make_renderer(name):
    if name == "gpu":
        import torch
        return GPURenderer("cuda" if torch.cuda.is_available() else "cpu")
    return CPURenderer()


# ====================================================================== 창 하나 생성
def _writer(path):
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (G.W, G.H))


def _dirs(root, ds):
    d = {k: os.path.join(root, ds, "train", k) for k in ("video", "audio", "labels", "meta")}
    for p in d.values(): os.makedirs(p, exist_ok=True)
    return d


def gen_window(job, renderer):
    """창 하나: main20 클립 1개(200프레임) + ctrl5 조각 4개(50프레임). 반환: 상태 dict."""
    t0 = time.time()
    row, rec, root = job["row"], job["rec"], job["root"]
    clip = row["clip"]; start = int(row["start_frame"]); mirror = int(row["mirror"])
    tr = FamTraj(json.loads(row["traj"]), 20.0); psis = json.loads(row["piece_psi"])
    pieces = [G.Traj("fixed", 5.0, float(p)) for p in psis]

    # --- 오디오와 피크 (피크 초과 창은 통째로 버린다)
    foa = G.read_foa(rec["wav"], start, N_MAIN)
    if mirror:
        foa = foa.copy(); foa[1] = -foa[1]                      # 거울: FOA 의 Y 채널 부호 반전
    t_s = np.arange(foa.shape[1]) / G.SR
    a_main = G.foa_to_stereo(foa, -tr.psi(t_s))
    seg = foa.shape[1] // 4
    a_pcs = [G.foa_to_stereo(foa[:, j * seg:(j + 1) * seg], -pieces[j].psi(np.arange(seg) / G.SR)) for j in range(4)]
    peak = max(float(np.abs(a_main).max()), *[float(np.abs(a).max()) for a in a_pcs])
    if peak > G.PEAK_LIMIT:
        return dict(status="peak", w=int(row["w"]), peak=peak)

    # --- 라벨
    lab = G.load_labels(rec["csv"])
    if mirror:
        lab = lab.copy(); lab["az"] = -lab["az"]                 # 거울: 세계 방위각 부호 반전
    pub_m, ext_m, trj_m = G.clip_labels(lab, start, N_MAIN, tr)
    parts = [G.clip_labels(lab, start + N_PIECE * j, N_PIECE, pieces[j]) for j in range(4)]
    assert len(pub_m) == sum(len(x[0]) for x in parts), "main20 과 ctrl5 라벨 행 수 불일치"

    dm, dc = _dirs(root, "main20"), _dirs(root, "ctrl5")
    names = [clip] + [f"{clip}_{j}" for j in range(4)]
    tmp = []
    def save(d, nm, a, pub, ext, trj):
        sf.write(os.path.join(d["audio"], nm + ".wav.tmp"), a.T, G.SR, format="WAV", subtype="PCM_16")
        pub.to_csv(os.path.join(d["labels"], nm + ".csv.tmp"), index=False)
        ext.to_csv(os.path.join(d["meta"], nm + "_ext.csv.tmp"), index=False)
        trj.to_csv(os.path.join(d["meta"], nm + "_traj.csv.tmp"), index=False)
    save(dm, names[0], a_main, pub_m, ext_m, trj_m)
    for j in range(4): save(dc, names[1 + j], a_pcs[j], *parts[j])

    # --- 영상: 원본 프레임을 한 번만 디코딩해서 main20 과 ctrl5 조각에 같이 쓴다
    cap = cv2.VideoCapture(rec["mp4"]); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = rec["fps"]
    t_abs = (start + np.arange(N_MAIN)) * 0.1 + 0.05
    idx = np.minimum(np.round(t_abs * fps).astype(int), total - 1)
    tt = G.frame_times(N_MAIN); psi_m, phi_m = tr.psi(tt), tr.phi(tt)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx[0])); cur = int(idx[0])
    wm = _writer(os.path.join(dm["video"], names[0] + ".tmp.mp4")); wp = None
    for k in range(N_MAIN):
        if k == 0 or idx[k] != idx[k - 1]:
            while cur < idx[k]:
                cap.grab(); cur += 1
            ok, frame = cap.read(); cur += 1
            if not ok: raise RuntimeError(f"read fail {rec['mp4']} frame {idx[k]}")
            if mirror: frame = cv2.flip(frame, 1)                # 거울: 360° 프레임 좌우 뒤집기
            renderer.set_frame(frame)
        wm.write(renderer.render(psi_m[k], phi_m[k]))
        j, kk = divmod(k, N_PIECE)
        if kk == 0: wp = _writer(os.path.join(dc["video"], names[1 + j] + ".tmp.mp4"))
        wp.write(renderer.render(float(psis[j]), 0.0))
        if kk == N_PIECE - 1: wp.release()
    wm.release(); cap.release()

    # --- 임시 이름 -> 최종 이름 (영상이 마지막 = 완료 표시)
    for d, nm in [(dm, names[0])] + [(dc, names[1 + j]) for j in range(4)]:
        for sub, fn in (("audio", nm + ".wav"), ("labels", nm + ".csv"), ("meta", nm + "_ext.csv"), ("meta", nm + "_traj.csv")):
            os.replace(os.path.join(d[sub], fn + ".tmp"), os.path.join(d[sub], fn))
        os.replace(os.path.join(d["video"], nm + ".tmp.mp4"), os.path.join(d["video"], nm + ".mp4"))   # 영상이 마지막 = 완료 표시
    return dict(status="ok", w=int(row["w"]), peak=peak, secs=time.time() - t0, rows=int(len(pub_m)))


# ====================================================================== 병렬 실행
_R = None

def _init_worker(renderer_name):
    global _R
    cv2.setNumThreads(1)
    _R = make_renderer(renderer_name)


def _work(job):
    try:
        return gen_window(job, _R)
    except Exception as e:
        return dict(status="fail", w=int(job["row"]["w"]), error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-400:])


def _manifest_rows(plan_rows, results):
    ok = {r["w"] for r in results if r["status"] == "ok"}
    m, c = [], []
    for _, r in plan_rows.iterrows():
        if int(r["w"]) not in ok: continue
        m.append({k: r[k] for k in ("clip", "recording", "room", "start_frame", "n_frames", "stage", "family", "mirror", "bonus", "rare_frames", "traj")} | {"split": "train"})
        for j, psi in enumerate(json.loads(r["piece_psi"])):
            c.append(dict(clip=f"{r['clip']}_{j}", split="train", recording=r["recording"], room=r["room"], start_frame=int(r["start_frame"]) + N_PIECE * j,
                          n_frames=N_PIECE, stage=r["stage"], family="fixed", mirror=r["mirror"], psi0=psi, parent_clip=r["clip"], j=j))
    return pd.DataFrame(m), pd.DataFrame(c)


def package(buf_shard, zip_dir_local, stage, shard):
    os.makedirs(zip_dir_local, exist_ok=True); out = {}
    for ds in ("main20", "ctrl5"):
        z = os.path.join(zip_dir_local, f"s{stage}_{ds}_train_p{shard:03d}.zip"); n = 0
        with zipfile.ZipFile(z + ".tmp", "w", zipfile.ZIP_STORED) as zf:
            base = os.path.join(buf_shard, ds)
            for r, _, fs in os.walk(base):
                for f in sorted(fs):
                    p = os.path.join(r, f); zf.write(p, os.path.relpath(p, base).replace("\\", "/")); n += 1
        os.replace(z + ".tmp", z); out[ds] = (z, n)
    return out


def upload_verify(local_zip, drive_dir, full=True):
    os.makedirs(drive_dir, exist_ok=True); dst = os.path.join(drive_dir, os.path.basename(local_zip))
    shutil.copyfile(local_zip, dst + ".part"); os.replace(dst + ".part", dst)
    if hasattr(os, "sync"): os.sync()
    rep = dict(zip=os.path.basename(local_zip), size_equal=os.path.getsize(dst) == os.path.getsize(local_zip))
    zl, zd = zipfile.ZipFile(local_zip), zipfile.ZipFile(dst)
    rep["central_dir_equal"] = [(i.filename, i.file_size, i.CRC) for i in zl.infolist()] == [(i.filename, i.file_size, i.CRC) for i in zd.infolist()]
    rep["testzip_bad"] = zd.testzip() if full else "skipped"
    rep["ok"] = bool(rep["size_equal"] and rep["central_dir_equal"] and rep["testzip_bad"] in (None, "skipped"))
    return rep


def run_generation(plan, valid_by_name, stage, drive_dir, buf, workers, renderer_name, shard_windows=250,
                   max_shards=0, full_verify=True, log=print):
    """묶음 단위: 생성 -> zip -> 드라이브 복사 -> 검증 -> 로컬 삭제. 끝난 묶음은 건너뛴다(이어 하기)."""
    import multiprocessing as mp
    os.makedirs(drive_dir, exist_ok=True)
    st_path = os.path.join(drive_dir, "status.json")
    status = json.load(open(st_path, encoding="utf8")) if os.path.exists(st_path) else {}
    nshard = math.ceil(len(plan) / shard_windows); todo = [s for s in range(nshard) if str(s) not in status or not status[str(s)].get("ok")]
    if max_shards: todo = todo[:max_shards]
    log(f"묶음 {nshard}개 중 남은 것 {len([s for s in range(nshard) if str(s) not in status])}개, 이번에 처리 {len(todo)}개 (묶음당 창 {shard_windows}개, 워커 {workers}, 렌더러 {renderer_name})")
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(renderer_name,)) as pool:
        for s in todo:
            t0 = time.time(); rows = plan.iloc[s * shard_windows:(s + 1) * shard_windows]
            bs = os.path.join(buf, f"s{stage}_p{s:03d}")
            if os.path.isdir(bs): shutil.rmtree(bs)
            jobs = [dict(row=r.to_dict(), rec={k: valid_by_name[r["recording"]][k] for k in ("wav", "mp4", "csv", "fps")}, root=bs) for _, r in rows.iterrows()]
            res = []
            for i, r in enumerate(pool.imap_unordered(_work, jobs, chunksize=1)):
                res.append(r)
                if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                    log(f"  묶음 {s}: {i+1}/{len(jobs)} ({time.time()-t0:.0f}초) 피크탈락 {sum(x['status']=='peak' for x in res)} 실패 {sum(x['status']=='fail' for x in res)}")
            fails = [x for x in res if x["status"] == "fail"]
            m, c = _manifest_rows(rows, res)
            zl = os.path.join(buf, "zips"); pk = package(bs, zl, stage, s)
            reps = []
            for ds, (z, n) in pk.items():
                reps.append(upload_verify(z, drive_dir, full=full_verify))
            for name, df in (("main20", m), ("ctrl5", c)):
                p = os.path.join(drive_dir, f"s{stage}_{name}_manifest_p{s:03d}.csv"); df.to_csv(p + ".part", index=False); os.replace(p + ".part", p)
            dropped = [x["w"] for x in res if x["status"] == "peak"]
            ok = all(r["ok"] for r in reps) and not fails
            status[str(s)] = dict(ok=ok, windows=len(rows), generated=len(m), dropped_peak=dropped, failed=[(f["w"], f["error"]) for f in fails],
                                  zips=reps, secs=round(time.time() - t0), main20_files=pk["main20"][1], ctrl5_files=pk["ctrl5"][1],
                                  mean_secs_per_window=round(float(np.mean([x["secs"] for x in res if x["status"] == "ok"])) if m is not None and len(m) else 0, 2))
            tmp = st_path + ".part"; json.dump(status, open(tmp, "w", encoding="utf8"), indent=1, ensure_ascii=False); os.replace(tmp, st_path)
            if ok:
                shutil.rmtree(bs); [os.remove(z) for z, _ in pk.values()]            # 검증이 끝난 묶음만 로컬에서 지운다
                log(f"묶음 {s} 완료: 클립 {len(m)}개+조각 {len(c)}개, 피크탈락 {len(dropped)}, {time.time()-t0:.0f}초 -> 로컬 삭제")
            else:
                log(f"묶음 {s} 에 문제가 있어 로컬 파일을 남겼습니다: 실패 {len(fails)}개, 검증 {[r['ok'] for r in reps]}")
                break
    return status


def merge_manifests(drive_dir, stage):
    """묶음별 매니페스트를 하나로 합친다(s{stage}_main20_manifest.csv, s{stage}_ctrl5_manifest.csv). ctrl5 쪽에 parent_clip, j, psi0 가 들어 있다."""
    out = {}
    for ds in ("main20", "ctrl5"):
        fs = sorted(glob.glob(os.path.join(drive_dir, f"s{stage}_{ds}_manifest_p*.csv")))
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
        df.to_csv(os.path.join(drive_dir, f"s{stage}_{ds}_manifest.csv"), index=False); out[ds] = int(len(df))
    return out


# ====================================================================== 자체 점검
def selftest(valid, split_of, sample_rec=None, log=print, gpu_device="cpu"):
    """필수 점검 (DATA_PLAN_v2.md 8절): 거울 3가지, 궤적 가족, GPU/CPU 렌더러 비교, 작은 실행(창 2개)."""
    import tempfile
    res = {}
    rec = next(p for p in valid if p["name"] == (sample_rec or "fold3_room6_mix001"))
    lab = G.load_labels(rec["csv"]); start = 800
    # (1) 거울 오디오 = 원래 오디오의 L/R 교환
    base = FamTraj(sample_traj("F1", 20.0, np.random.default_rng(1)), 20.0)
    foa = G.read_foa(rec["wav"], start, N_MAIN); t_s = np.arange(foa.shape[1]) / G.SR
    a0 = G.foa_to_stereo(foa, -base.psi(t_s))
    fm = foa.copy(); fm[1] = -fm[1]
    a1 = G.foa_to_stereo(fm, -NegTraj(base).psi(t_s))
    res["mirror_audio_max_err"] = float(np.abs(a1 - a0[::-1]).max())
    # (2) 거울 영상 = 원래 렌더링의 좌우 뒤집기
    cap = cv2.VideoCapture(rec["mp4"]); cap.set(cv2.CAP_PROP_POS_FRAMES, 800); ok, fr = cap.read(); cap.release()
    rd = G.Renderer(); diffs = []
    for psi, phi in ((0, 0), (100, -10), (-150, -20), (37, -5)):
        orig = rd.render(fr, psi, phi); mir = rd.render(cv2.flip(fr, 1), -psi, phi)
        diffs.append(float(np.abs(mir.astype(float) - orig[:, ::-1].astype(float)).mean()))
    res["mirror_video_mean_abs_diff"] = max(diffs)
    # (3) 거울 라벨: 방위각 부호 반전, onscreen 불변
    p0, e0, _ = G.clip_labels(lab, start, N_MAIN, base)
    lm = lab.copy(); lm["az"] = -lm["az"]
    p1, e1, _ = G.clip_labels(lm, start, N_MAIN, NegTraj(base))
    wrapd = ((p1.azimuth.values + p0.azimuth.values + 180) % 360) - 180
    res["mirror_label_az_sign_err"] = int(np.abs(wrapd).max()) if len(p0) else 0
    res["mirror_label_onscreen_equal"] = bool((p0.onscreen.values == p1.onscreen.values).all())
    # (4) 궤적 가족: 각속도 상한, 피치 범위, 소리 연속
    rng = np.random.default_rng(7); fam_rate = {}; pitch_ok = True
    for fam in DEFAULT_MIX:
        rates = []
        for _ in range(300):
            tj = FamTraj(sample_traj(fam, 20.0, rng), 20.0); rates.append(max_rate(tj))
            ph = tj.phi(np.arange(0, 20, 0.05)); pitch_ok &= bool(ph.min() >= -25.001 and ph.max() <= 0.001)
        fam_rate[fam] = round(max(rates), 1)
    res["family_max_rate_deg_per_s"] = fam_rate; res["pitch_in_range"] = pitch_ok
    worst = FamTraj(sample_traj("F3", 20.0, np.random.default_rng(3)), 20.0)
    ref = max(np.abs(np.diff(G.foa_to_stereo(foa, 0.0), axis=1)).max(), np.abs(np.diff(G.foa_to_stereo(foa, 90.0), axis=1)).max())
    res["audio_jump_ratio_F3"] = float(np.abs(np.diff(G.foa_to_stereo(foa, -worst.psi(t_s)), axis=1)).max() / ref)
    # (5) GPU 렌더러 vs CPU 렌더러
    try:
        gr = GPURenderer(gpu_device); gr.set_frame(fr); d = []
        for psi, phi in ((0, 0), (100, -10), (-150, -20), (37, -5), (179, -25)):
            d.append(float(np.abs(gr.render(psi, phi).astype(float) - rd.render(fr, psi, phi).astype(float)).mean()))
        res["gpu_vs_cpu_mean_abs_diff"] = max(d); res["gpu_device"] = gr.dev
    except Exception as e:
        res["gpu_vs_cpu_mean_abs_diff"] = None; res["gpu_error"] = f"{type(e).__name__}: {e}"
    # (6) 작은 실행: 창 2개 (거울 1개 포함)
    tmp = tempfile.mkdtemp(); plan, _ = make_plan(valid, split_of, 0, 12)
    rows = plan.head(2).copy(); rows.loc[rows.index[0], "mirror"] = 1; rows.loc[rows.index[1], "mirror"] = 0
    vb = {p["name"]: p for p in valid}; rr = CPURenderer(); out = []
    for _, r in rows.iterrows():
        job = dict(row=r.to_dict(), rec={k: vb[r["recording"]][k] for k in ("wav", "mp4", "csv", "fps")}, root=tmp)
        out.append(gen_window(job, rr))
    chk = []
    for _, r in rows.iterrows():
        c = r["clip"]; cap = cv2.VideoCapture(os.path.join(tmp, "main20", "train", "video", c + ".mp4")); nf = int(cap.get(7)); cap.release()
        wi = sf.info(os.path.join(tmp, "main20", "train", "audio", c + ".wav"))
        pc = []
        for j in range(4):
            cap = cv2.VideoCapture(os.path.join(tmp, "ctrl5", "train", "video", f"{c}_{j}.mp4")); pc.append(int(cap.get(7))); cap.release()
            wj = sf.info(os.path.join(tmp, "ctrl5", "train", "audio", f"{c}_{j}.wav")); pc.append(wj.frames)
        lm_ = pd.read_csv(os.path.join(tmp, "main20", "train", "labels", c + ".csv"))
        lp_ = sum(len(pd.read_csv(os.path.join(tmp, "ctrl5", "train", "labels", f"{c}_{j}.csv"))) for j in range(4))
        chk.append(dict(clip=c, main_frames=nf, main_samples=wi.frames, piece_frames_samples=pc, label_rows_equal=len(lm_) == lp_, status=[o["status"] for o in out]))
    res["tiny_run"] = chk; shutil.rmtree(tmp, ignore_errors=True)
    res["PASS"] = bool(res["mirror_audio_max_err"] < 1e-9 and res["mirror_video_mean_abs_diff"] < 1.0 and res["mirror_label_az_sign_err"] <= 1
                       and res["mirror_label_onscreen_equal"] and all(v <= MAX_RATE for v in fam_rate.values()) and pitch_ok
                       and res["audio_jump_ratio_F3"] < 1.5
                       and all(c["main_frames"] == 200 and c["main_samples"] == 480000 and c["piece_frames_samples"] == [50, 120000] * 4 and c["label_rows_equal"] for c in chk)
                       and all(s == "ok" for c in chk for s in c["status"]))
    log(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["selftest", "plan"]); ap.add_argument("--stage", type=int, default=1); ap.add_argument("--n", type=int, default=6000)
    a = ap.parse_args()
    valid = load_valid(); split_of, info = G.make_split(valid)
    if a.cmd == "selftest": selftest(valid, split_of)
    else:
        plan, meta = make_plan(valid, split_of, a.stage, a.n); print(meta); print(json.dumps(plan_summary(plan, valid, split_of), ensure_ascii=False, indent=1))
