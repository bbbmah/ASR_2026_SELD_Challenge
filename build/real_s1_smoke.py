# [변경 이력 시작]
#   2026-10-07  최초 생성: 드라이브의 실제 S1 zip 일부로 이름 규칙, 특징 추출, 데이터 로더 모양을 확인
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
import os, sys, zipfile, shutil, re, json, time
sys.path.insert(0, "G:/내 드라이브/Colab Notebooks/ASR_2026-2/for_dataset/baseline/seld_challenge")
import config as C
DR = "G:/내 드라이브/Colab Notebooks/ASR_2026-2/for_dataset"
tmp = "C:/asrwork/real_s1_smoke"; shutil.rmtree(tmp, ignore_errors=True)
for ds, nfr, nclip in (("main20", 200, 3), ("ctrl5", 50, 4)):
    zp = f"{DR}/generated/s1/s1_{ds}_train_p000.zip"; t0 = time.time()
    z = zipfile.ZipFile(zp); names = z.namelist()
    mp4 = sorted(n for n in names if n.endswith(".mp4"))[:nclip]
    print(f"[{ds}] zip 항목 {len(names)}개 (읽기 {time.time()-t0:.0f}초), 영상 {sum(n.endswith('.mp4') for n in names)}개, 첫 항목 {names[0]}")
    ok_pat = all(re.match(r"^train/(video|audio|labels|meta)/train_s1_\d{5}(_\d)?(\.mp4|\.wav|\.csv|_ext\.csv|_traj\.csv)$", n) for n in names)
    print("   이름 규칙 일치:", ok_pat)
    for m in mp4:
        base = os.path.basename(m)[:-4]
        for sub, fn in (("video", base + ".mp4"), ("audio", base + ".wav"), ("labels", base + ".csv"), ("meta", base + "_ext.csv"), ("meta", base + "_traj.csv")):
            z.extract(f"train/{sub}/{fn}", f"{tmp}/{ds}")
print("추출 완료")
cfg = C.load_config("main20", "B2", [f"drive_root={DR}", f"data_root={tmp}", f"out_root={tmp}/out", "train_stages=0,1", "num_workers=0"])
import extract_features as EF, torch
print("clip_names(단계1):", EF.clip_names(f"{tmp}/main20/train/audio", ".wav", "train", 1), "| 단계0:", EF.clip_names(f"{tmp}/main20/train/audio", ".wav", "train", 0))
for ds in ("main20", "ctrl5"):
    p = dict(cfg.params); p["label_sequence_length"] = C.LABEL_LEN[ds]
    ex = EF.SELDFeatureExtractor(p, C.dataset_dirs(cfg, ds), C.feat_dir(cfg, ds)); t0 = time.time()
    ex.extract_features("train", stage=1)
    d = torch.load(f"{tmp}/out/features/{ds}/audio/s1train_000.pt"); v = torch.load(f"{tmp}/out/features/{ds}/video_center/s1train_000.pt"); l = torch.load(f"{tmp}/out/features/{ds}/labels_adpit/s1train_000.pt")
    print(f"[{ds}] 특징 추출 {time.time()-t0:.0f}초 | 이름 {d['names']} | 오디오 {tuple(d['data'].shape)} 영상 {tuple(v['data'].shape)} 라벨 {tuple(l['data'].shape)}")
# 데이터 로더로 읽기
from data_generator import DataGenerator
for ds in ("main20", "ctrl5"):
    p = dict(cfg.params); p["label_sequence_length"] = C.LABEL_LEN[ds]
    g = DataGenerator(p, C.feat_dir(cfg, ds), "train", stages=(1,)); (a, v), lab = g[0]
    print(f"[{ds}] DataGenerator {len(g)}개, 첫 항목 오디오 {tuple(a.shape)} 영상 {tuple(v.shape)} 라벨 {tuple(lab.shape)}, 메모리 {g.ram_gb():.3f}GB, 단계별 {g.stage_counts}")
