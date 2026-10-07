# [변경 이력 시작]
#   2026-10-02  최초 생성: 카메라 궤적, 영상 렌더러(투시 변환), FOA에서 스테레오 오디오 회전, 라벨 변환, 분할(split). 이후 변경 없음(다른 스크립트가 그대로 가져다 씀)
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""챌린지 데이터셋 생성 모듈: 궤적, 렌더링, 오디오 회전, 라벨 변환, 클립/스플릿."""
import os, glob, math, json
import numpy as np
import pandas as pd
import cv2
import soundfile as sf

cv2.setNumThreads(1)

RAW = os.environ.get("RAW", r"C:\asrwork\raw")
SR = 24000
FOV_H = 100.0
W, H = 640, 360
TAN_H = math.tan(math.radians(FOV_H / 2))
TAN_V = TAN_H * H / W                      # tan(33.84deg)
FOV_V = 2 * math.degrees(math.atan(TAN_V))
PEAK_LIMIT = 0.9999
MIN_LABELED = 30
TRAIN_ROOMS = [4, 6, 7, 9, 12, 13, 14, 21, 22]
TAMPERE = {4, 6, 7, 9, 12, 13, 14}
SONY = {21, 22}


# ---------------------------------------------------------------- 녹음 목록
def list_recordings():
    recs = []
    for c in sorted(glob.glob(f"{RAW}/metadata_dev/*/*.csv")):
        name = os.path.splitext(os.path.basename(c))[0]
        sub = os.path.basename(os.path.dirname(c))
        fold = int(name.split("_")[0][4:])
        room = int(name.split("_")[1][4:])
        recs.append(dict(name=name, sub=sub, fold=fold, room=room, csv=c,
                         wav=f"{RAW}/foa_dev/{sub}/{name}.wav",
                         mp4=f"{RAW}/video_dev/{sub}/{name}.mp4"))
    return recs


def probe(rec):
    """오디오/영상/라벨 길이(초)와 프레임 정보."""
    out = dict(rec)
    out["has_wav"] = os.path.exists(rec["wav"])
    out["has_mp4"] = os.path.exists(rec["mp4"])
    lab = pd.read_csv(rec["csv"], header=None)
    out["label_end_s"] = (lab[0].max() + 1) * 0.1
    out["labeled_frames"] = int(lab[0].nunique())
    if out["has_wav"]:
        i = sf.info(rec["wav"])
        out["sr"] = i.samplerate; out["channels"] = i.channels
        out["audio_s"] = i.frames / i.samplerate
    if out["has_mp4"]:
        cap = cv2.VideoCapture(rec["mp4"])
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT); fps = cap.get(cv2.CAP_PROP_FPS)
        out["video_frames"] = int(n); out["fps"] = fps; out["video_s"] = n / fps
        out["video_w"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); out["video_h"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    if out["has_wav"] and out["has_mp4"]:
        out["len_frames"] = int(math.floor(min(out["audio_s"], out["video_s"]) * 10 + 1e-9))
    return out


# ---------------------------------------------------------------- 궤적
def wrap180(x):
    return (np.asarray(x, dtype=np.float64) + 180.0) % 360.0 - 180.0


class Traj:
    """kind='sweep'(main20) 또는 'fixed'(ctrl5). psi/phi는 도 단위, t는 클립 시작 기준 초."""
    def __init__(self, kind, clip_s, psi0, s=1, omega=0.0, P=15.0, phi0=0.0):
        self.kind, self.clip_s = kind, clip_s
        self.psi0, self.s, self.omega, self.P, self.phi0 = psi0, s, omega, P, phi0

    @staticmethod
    def sample(kind, clip_s, rng):
        if kind == "sweep":
            return Traj("sweep", clip_s, rng.uniform(0, 360), int(rng.choice([-1, 1])),
                        360.0 / clip_s * rng.uniform(0.9, 1.1), rng.uniform(10, 20), rng.uniform(0, 2 * np.pi))
        return Traj("fixed", clip_s, float(rng.integers(0, 360)))

    def psi(self, t):
        t = np.asarray(t, dtype=np.float64)
        if self.kind == "fixed":
            return np.full_like(t, self.psi0)
        return self.psi0 + self.s * self.omega * t

    def phi(self, t):
        t = np.asarray(t, dtype=np.float64)
        if self.kind == "fixed":
            return np.zeros_like(t)
        return -12.5 + 12.5 * np.sin(2 * np.pi * t / self.P + self.phi0)

    def params(self):
        return dict(kind=self.kind, psi0=round(self.psi0, 4), s=self.s, omega=round(self.omega, 4),
                    P=round(self.P, 4), phi0=round(self.phi0, 4))

    @staticmethod
    def from_params(clip_s, p):
        return Traj(p["kind"], clip_s, p["psi0"], p["s"], p["omega"], p["P"], p["phi0"])


def frame_times(n):
    return 0.1 * np.arange(n) + 0.05


# ---------------------------------------------------------------- 좌표
def cam_axes(psi_deg, phi_deg):
    """카메라 앞/왼/위 단위벡터(세계 좌표, x앞 y왼 z위). 반환 (...,3) 각각."""
    p, f = np.radians(psi_deg), np.radians(phi_deg)
    fwd = np.stack([np.cos(f) * np.cos(p), np.cos(f) * np.sin(p), np.sin(f)], -1)
    left = np.stack([-np.sin(p), np.cos(p), np.zeros_like(p)], -1)
    up = np.stack([-np.sin(f) * np.cos(p), -np.sin(f) * np.sin(p), np.cos(f)], -1)
    return fwd, left, up


def dir_vec(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], -1)


def to_cam(az, el, psi, phi):
    """세계 방향(az, el) -> 카메라 좌표 (x앞, y왼, z위)."""
    d = dir_vec(az, el)
    fwd, left, up = cam_axes(np.asarray(psi, dtype=np.float64), np.asarray(phi, dtype=np.float64))
    return np.stack([(d * fwd).sum(-1), (d * left).sum(-1), (d * up).sum(-1)], -1)


def onscreen(az, el, psi, phi):
    c = to_cam(az, el, psi, phi)
    x, y, z = c[..., 0], c[..., 1], c[..., 2]
    return ((x > 0) & (np.abs(y) <= x * TAN_H) & (np.abs(z) <= x * TAN_V)).astype(int)


# ---------------------------------------------------------------- 영상 렌더링
class Renderer:
    def __init__(self, in_hw=(960, 1920)):
        self.ih, self.iw = in_hw
        u = (np.arange(W) + 0.5) / W
        v = (np.arange(H) + 0.5) / H
        ry = (1 - 2 * u)[None, :] * TAN_H * np.ones((H, 1))      # 왼쪽 +
        rz = (1 - 2 * v)[:, None] * TAN_V * np.ones((1, W))      # 위 +
        rx = np.ones((H, W))
        self.rays = np.stack([rx, ry, rz], -1).reshape(-1, 3)    # 카메라 좌표 광선

    def maps(self, psi, phi):
        fwd, left, up = cam_axes(np.float64(psi), np.float64(phi))
        M = np.stack([fwd, left, up], 1)                          # 열 = 카메라 축(세계 좌표)
        d = self.rays @ M.T
        az = np.arctan2(d[:, 1], d[:, 0])
        el = np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1]))
        mx = (0.5 - az / (2 * np.pi)) * self.iw - 0.5
        my = (0.5 - el / np.pi) * self.ih - 0.5
        return (mx.reshape(H, W).astype(np.float32), my.reshape(H, W).astype(np.float32))

    def render(self, equi, psi, phi):
        mx, my = self.maps(psi, phi)
        return cv2.remap(equi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# ---------------------------------------------------------------- 오디오
def foa_to_stereo(foa, theta_deg):
    """foa: (4,N) W,Y,Z,X. theta_deg: 스칼라 또는 (N,). 생성기 변환과 동일."""
    Wc, Y, X = foa[0], foa[1], foa[3]
    r = np.radians(theta_deg) * np.ones_like(Wc)
    newY = np.sin(r) * X + np.cos(r) * Y
    return np.stack([Wc + newY, Wc - newY])


def read_foa(wav, start_frame, n_frames):
    a = start_frame * SR // 10
    b = (start_frame + n_frames) * SR // 10
    x, sr = sf.read(wav, start=a, stop=b, dtype="float64", always_2d=True)
    assert sr == SR
    return x.T


def clip_audio(wav, start_frame, n_frames, traj):
    foa = read_foa(wav, start_frame, n_frames)
    t = np.arange(foa.shape[1]) / SR
    return foa_to_stereo(foa, -traj.psi(t))


# ---------------------------------------------------------------- 라벨
def load_labels(csv):
    return pd.read_csv(csv, header=None, names=["frame", "class", "source", "az", "el", "dist"])


def clip_labels(lab, start_frame, n_frames, traj):
    cut = lab[(lab.frame >= start_frame) & (lab.frame < start_frame + n_frames)].copy()
    cut["frame"] -= start_frame
    t = frame_times(n_frames)
    psi, phi = traj.psi(t), traj.phi(t)
    k = cut.frame.values
    p, f = psi[k], phi[k]
    az_new = wrap180(cut.az.values - p)
    on = onscreen(cut.az.values.astype(float), cut.el.values.astype(float), p, f)
    ext = pd.DataFrame(dict(frame=cut.frame.values, **{"class": cut["class"].values}, source=cut.source.values,
                            azimuth=np.round(az_new).astype(int), distance=cut.dist.values, onscreen=on,
                            az_world=cut.az.values, el_world=cut.el.values,
                            cam_yaw=np.round(wrap180(p), 3), cam_pitch=np.round(f, 3)))
    pub = ext[["frame", "class", "source", "azimuth", "distance", "onscreen"]]
    trj = pd.DataFrame(dict(frame=np.arange(n_frames), psi=np.round(wrap180(psi), 3), phi=np.round(phi, 3)))
    return pub, ext, trj


# ---------------------------------------------------------------- 영상 생성
def write_video(mp4_in, fps_in, start_frame, n_frames, traj, out_path):
    cap = cv2.VideoCapture(mp4_in)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t_abs = (start_frame + np.arange(n_frames)) * 0.1 + 0.05
    idx = np.minimum(np.round(t_abs * fps_in).astype(int), total - 1)
    t = frame_times(n_frames)
    psi, phi = traj.psi(t), traj.phi(t)
    rd = Renderer()
    wr = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx[0]))
    cur = int(idx[0]); frame = None
    for k in range(n_frames):
        if k == 0 or idx[k] != idx[k - 1]:
            while cur < idx[k]:
                cap.grab(); cur += 1
            ok, frame = cap.read(); cur += 1
            if not ok:
                raise RuntimeError(f"read fail {mp4_in} frame {idx[k]}")
        wr.write(rd.render(frame, psi[k], phi[k]))
    wr.release(); cap.release()


# ---------------------------------------------------------------- 스플릿
def make_split(probed):
    """probed: 유효 녹음 probe 리스트. test=fold4, val=학습 방 중 Tampere1+Sony1."""
    df = pd.DataFrame(probed)
    tr = df[df.fold == 3]
    lab = tr.groupby("room").labeled_frames.sum()
    total = lab.sum()
    best = None
    for a in TAMPERE:
        for b in SONY:
            if a in lab.index and b in lab.index:
                r = (lab[a] + lab[b]) / total
                score = abs(r - 0.175)
                if best is None or score < best[0]:
                    best = (score, a, b, r)
    _, a, b, r = best
    split = {}
    for p in probed:
        if p["fold"] == 4: split[p["name"]] = "test"
        elif p["room"] in (a, b): split[p["name"]] = "val"
        else: split[p["name"]] = "train"
    info = dict(val_rooms=[int(a), int(b)], val_label_fraction_of_train_rooms=float(r),
                label_frames_per_room={int(k): int(v) for k, v in lab.items()},
                test_rooms=sorted({int(p["room"]) for p in probed if p["fold"] == 4}),
                train_rooms=sorted({int(p["room"]) for p in probed if p["fold"] == 3 and p["room"] not in (a, b)}))
    return split, info
