# [변경 이력 시작]
#   2026-10-03  최초 생성: 소규모 시험용 미니 데이터와 zip 만들기(데이터셋마다 train 4, val 2, test 2)
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""소규모 시험용 미니 데이터: 데이터셋마다 train 4, val 2, test 2(val 클립을 다른 이름으로 복사, 가짜 비공개 정답). 학습 결과 품질과 무관하게 코드가 끝까지 도는지만 본다."""
import os, zipfile, shutil
DS = r"C:\asrwork\dataset"; OUT = r"C:\asrwork\mini"
shutil.rmtree(OUT, ignore_errors=True)
PUB = os.path.join(OUT, "pub"); INT = os.path.join(OUT, "internal")
os.makedirs(PUB); os.makedirs(INT)

def files(ds, sp, clip, with_labels=True, with_meta=True):
    base = os.path.join(DS, ds, sp)
    out = [("video", clip + ".mp4"), ("audio", clip + ".wav")]
    if with_labels: out.append(("labels", clip + ".csv"))
    if with_meta: out += [("meta", clip + "_ext.csv"), ("meta", clip + "_traj.csv")]
    return [(os.path.join(base, d, f), f"{sp}/{d}/{f}") for d, f in out]

def zipit(path, pairs):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        for src, arc in pairs: z.write(src, arc)

for ds, splits in (("main20", None), ("ctrl5", None), ("fix20", ("val", "test"))):
    out_dir = PUB if ds == "main20" else INT
    name = lambda sp: {"train": "train.zip", "val": "val.zip", "test": "test_inputs.zip"}[sp] if ds == "main20" else f"internal_{ds}_{sp}.zip"
    train = [f"train_{i:05d}" for i in range(4)]
    val = [f"val_{i:05d}" for i in range(2)]
    test_src = [f"val_{i:05d}" for i in (2, 3)]
    if ds != "fix20":
        zipit(os.path.join(out_dir, name("train")), [p for c in train for p in files(ds, "train", c)])
    zipit(os.path.join(out_dir, name("val")), [p for c in val for p in files(ds, "val", c)])
    # test 입력: val 클립 2, 3을 test_0000x 이름으로
    tp, pp = [], []
    for k, c in enumerate(test_src):
        t = f"test_{k:05d}"
        for sub, ext in (("video", ".mp4"), ("audio", ".wav")):
            tp.append((os.path.join(DS, ds, "val", sub, c + ext), f"test/{sub}/{t}{ext}"))
        pp.append((os.path.join(DS, ds, "val", "labels", c + ".csv"), f"private/test_labels/{t}.csv"))
    zipit(os.path.join(out_dir, name("test")), tp)
    pname = "private_test.zip" if ds == "main20" else f"internal_{ds}_private.zip"
    if ds == "ctrl5":     # 제외 목록 시험용 가짜 부모 대응: val_00001 의 부모를 val_00088 로 둔다
        pm = os.path.join(OUT, "parent_map.csv")
        open(pm, "w").write("clip,parent_clip,j,start_frame,recording,psi\nval_00001,val_00088,0,0,x,0\n")
        pp.append((pm, "private/parent_map.csv"))
    zipit(os.path.join(out_dir, pname), pp)
for r, _, fs in os.walk(OUT):
    for f in fs: print(os.path.relpath(os.path.join(r, f), OUT), os.path.getsize(os.path.join(r, f)))
