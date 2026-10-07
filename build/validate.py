# [변경 이력 시작]
#   2026-10-02  최초 생성: 필수 검증 5개(입력 확인, 좌표 단위 시험, 렌더링 비교, 오디오 회전 시험, onscreen 비교)
#   2026-10-02  그림 저장을 cv2.imencode로 변경(한글 경로에서 imwrite가 조용히 실패함)
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""필수 검증 5개. 사용: python validate.py [1 2 3 4 5]"""
import sys, os, json, math
import numpy as np, pandas as pd, cv2, soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import dsgen as G

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "validation"); os.makedirs(OUT, exist_ok=True)
WORK = r"C:\asrwork"
GEN_DIR = os.path.join(HERE, "..", "baseline", "dcase2025_stereo_seld_data_generator")
sys.path.insert(0, GEN_DIR)
results = {}


def imsave(path, img):
    cv2.imencode('.png', img)[1].tofile(path)   # 한글 경로 대응



def v1():
    recs = G.list_recordings()
    with ThreadPoolExecutor(8) as ex:
        pr = list(ex.map(G.probe, recs))
    df = pd.DataFrame(pr)
    n_wav = sum(p["has_wav"] for p in pr); n_mp4 = sum(p["has_mp4"] for p in pr)
    bad = []
    for p in pr:
        why = []
        if not p["has_wav"]: why.append("no_wav")
        if not p["has_mp4"]: why.append("no_video")
        if p["has_wav"] and p["has_mp4"]:
            if abs(p["audio_s"] - p["video_s"]) >= 5: why.append(f"audio-video {p['audio_s']-p['video_s']:.1f}s")
            if p["label_end_s"] - min(p["audio_s"], p["video_s"]) >= 5: why.append("label beyond media")
            if p["sr"] != G.SR or p["channels"] != 4: why.append(f"sr/ch {p['sr']}/{p['channels']}")
            if (p["video_w"], p["video_h"]) != (1920, 960): why.append("video size")
        p["exclude"] = ";".join(why)
        if why: bad.append((p["name"], p["exclude"]))
    df.to_csv(os.path.join(OUT, "input_check.csv"), index=False)
    valid = [p for p in pr if not p["exclude"]]
    json.dump(valid, open(os.path.join(WORK, "probe_valid.json"), "w"), default=float)
    names = [p["name"] for p in pr]
    ok = len(pr) == 168 and n_mp4 == 156 and n_wav == 168
    print(f"labels={len(pr)} wav={n_wav} video={n_mp4} valid={len(valid)}")
    print("excluded:", bad)
    lens = df.dropna(subset=["audio_s", "video_s"])
    print("max |audio-video| s:", (lens.audio_s - lens.video_s).abs().max(), " fps:", lens.fps.min(), lens.fps.max())
    print("label_end - min(media) range:", (lens.label_end_s - lens[["audio_s", "video_s"]].min(axis=1)).describe()[["min", "max"]].to_dict())
    print("V1", "PASS" if ok else "FAIL")


def v2():
    # (a) 좌표 단위 시험
    rng = np.random.default_rng(0); okall = True
    for _ in range(5):
        psi, phi = rng.uniform(-180, 180), rng.uniform(-40, 40)
        c = G.to_cam(psi, phi, psi, phi)
        sw = G.to_cam(phi, psi, psi, phi)        # az/el 바꿔 넣기
        lr = G.to_cam(-psi, phi, psi, phi)       # 좌우 뒤집기
        ud = G.to_cam(psi, -phi, psi, phi)       # 위아래 뒤집기
        okall &= np.allclose(c, [1, 0, 0], atol=1e-9)
        # 부호 확인: 왼쪽(+az)으로 5도 -> y>0, 위(+el)로 5도 -> z>0
        l = G.to_cam(psi + 5, phi, psi, phi); u = G.to_cam(psi, phi + 5, psi, phi)
        okall &= l[1] > 0 and u[2] > 0 and abs(l[2]) < 0.2 and abs(u[1]) < 1e-9
    print("front=(1,0,0) & sign(left=+y, up=+z):", okall)
    print("  example psi=30,phi=-10: az=psi,el=phi ->", np.round(G.to_cam(30, -10, 30, -10), 6),
          "| az/el swapped ->", np.round(G.to_cam(-10, 30, 30, -10), 3),
          "| az sign flipped ->", np.round(G.to_cam(-30, -10, 30, -10), 3))
    # (b) 그림 1장
    name = "fold3_room6_mix001"
    p = [q for q in json.load(open(os.path.join(WORK, "probe_valid.json"))) if q["name"] == name][0]
    lab = G.load_labels(p["csv"]); row = lab[(lab.frame == 838) & (lab["class"] == 9) & (lab.source == 2)]
    print(row.to_string(index=False))
    az, el = float(row.az.iloc[0]), float(row.el.iloc[0])
    cap = cv2.VideoCapture(p["mp4"]); idx = int(round((838 * 0.1 + 0.05) * p["fps"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, fr = cap.read(); cap.release()
    img = G.Renderer().render(fr, az, el)
    cv2.line(img, (G.W // 2 - 12, G.H // 2), (G.W // 2 + 12, G.H // 2), (0, 0, 255), 1)
    cv2.line(img, (G.W // 2, G.H // 2 - 12), (G.W // 2, G.H // 2 + 12), (0, 0, 255), 1)
    eq = cv2.resize(fr, (960, 480)); imsave(os.path.join(OUT, "fig1_check2_center.png"), np.vstack([cv2.resize(img, (960, 540)), eq]))
    print("saved fig1_check2_center.png (위: 렌더링, 십자가=소리 방향; 아래: 원본 등장방형)")
    print("V2 numeric", "PASS" if okall else "FAIL", "(그림은 눈으로 확인)")


def v3():
    import utils as GU                      # 생성기 utils (E2PFast)
    name = "fold3_room6_mix001"
    p = [q for q in json.load(open(os.path.join(WORK, "probe_valid.json"))) if q["name"] == name][0]
    cap = cv2.VideoCapture(p["mp4"]); cap.set(cv2.CAP_PROP_POS_FRAMES, 800); ok, fr = cap.read(); cap.release()
    rd = G.Renderer(); diffs = []
    for psi in (0, 100, -150):
        mine = rd.render(fr, psi, 0)
        e2p = GU.E2PFast(in_hw=(960, 1920), fov_deg=(100, G.FOV_V), u_deg=-psi, v_deg=0, out_hw=(360, 640))
        ref = e2p.e2p(fr).astype(np.uint8)
        d = np.abs(mine.astype(float) - ref.astype(float)).mean(); diffs.append(d)
        imsave(os.path.join(OUT, f"fig_v3_psi{psi}.png"), np.hstack([mine, ref]))
        print(f"psi={psi}: mean abs diff = {d:.3f}/255")
    print("V3", "PASS" if max(diffs) <= 5 else "FAIL")


def gen_stereo(foa, deg):                    # 생성기 make_audio_video 안의 변환 그대로 복사
    W_, Y, X = foa[0], foa[1], foa[3]
    rad = np.ones(len(W_)) * deg / 180 * np.pi
    newY = np.multiply(np.sin(rad), X) + np.multiply(np.cos(rad), Y)
    return np.stack([W_ + newY, W_ - newY])


def v4():
    name = "fold3_room6_mix001"
    p = [q for q in json.load(open(os.path.join(WORK, "probe_valid.json"))) if q["name"] == name][0]
    foa = G.read_foa(p["wav"], 800, 200)
    errs = []
    for deg in (0, 37, -120, 250):
        tr = G.Traj("fixed", 20, -deg)       # theta=-psi=deg
        errs.append(np.abs(G.clip_audio(p["wav"], 800, 200, tr) - gen_stereo(foa, deg)).max())
    print("(a) const psi max err:", max(errs))
    tr = G.Traj("sweep", 20, 10.0, 1, 18.0, 15, 0)
    out = G.clip_audio(p["wav"], 800, 200, tr)
    ref = gen_stereo(foa, 0)
    ref_jump = max(np.abs(np.diff(ref, axis=1)).max(), np.abs(np.diff(gen_stereo(foa, 90), axis=1)).max())
    out_jump = np.abs(np.diff(out, axis=1)).max()
    print(f"(b) max adjacent diff: ours={out_jump:.5f}, source(θ=0/90)={ref_jump:.5f}, ratio={out_jump/ref_jump:.3f}")
    print("V4", "PASS" if max(errs) < 1e-6 and out_jump <= 1.5 * ref_jump else "FAIL")


def v5():
    from utils import CheckOnOffFast
    cwd = os.getcwd(); os.chdir(GEN_DIR); chk = CheckOnOffFast(); os.chdir(cwd)
    name = "fold3_room6_mix001"
    lab = G.load_labels([q for q in json.load(open(os.path.join(WORK, "probe_valid.json"))) if q["name"] == name][0]["csv"])
    rng = np.random.default_rng(1); agree = 0; tot = 0
    for deg in rng.integers(0, 360, 6):
        for r in lab.itertuples():
            ref = chk.check(int(deg), int(r.az), int(r.el))[0]
            mine = int(G.onscreen(float(r.az), float(r.el), -float(deg), 0.0))
            agree += int(ref == mine); tot += 1
    print(f"agreement {agree}/{tot} = {agree/tot:.4f}")
    print("V5", "PASS" if agree / tot >= 0.98 else "FAIL")


if __name__ == "__main__":
    for k in (sys.argv[1:] or ["1", "2", "3", "4", "5"]):
        print(f"===== 검증 {k} =====")
        {"1": v1, "2": v2, "3": v3, "4": v4, "5": v5}[k]()
