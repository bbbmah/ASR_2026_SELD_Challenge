# [변경 이력 시작]
#   2026-10-06  최초 생성: 드라이브 사본이 불완전한 ctrl5_B2 val 예측을 저장된 최고 모델로 로컬에서 다시 만들고 Colab 점수와 대조
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""ctrl5_B2 val 예측 csv 재생성 (드라이브 사본이 불완전해서). 드라이브 결과는 건드리지 않고 로컬(C:\asrwork\recheck)에 쓴다."""
import sys, os, json
sys.path.insert(0, r"G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset\baseline\seld_challenge")
import torch, config as C, run
from data_generator import DataGenerator
from score import score_dirs
DR = r"G:/내 드라이브/Colab Notebooks/ASR_2026-2/for_dataset"
cfg = C.load_config("ctrl5", "B2", [f"drive_root={DR}", "data_root=C:/asrwork/dataset", "num_workers=0"])
print(cfg.run_name, cfg.run_dir, cfg.batch_size_eval)
dev = "cpu"; torch.manual_seed(1)
p = run.params_for(cfg, "ctrl5")
model, ck = run.load_best(cfg, dev); print("best epoch", ck["epoch"] + 1, "saved F", ck["best_value"])
ds = DataGenerator(p, C.feat_dir(cfg, "ctrl5"), "val")
out = r"C:\asrwork\recheck\ctrl5_B2_s1\pred_val_real"
run.predict_split(model, ds, cfg.batch_size_eval, 0, p, out, dev)
print("예측 csv:", len(os.listdir(out)))
r = score_dirs(out, r"C:\asrwork\dataset\ctrl5\val\labels", 50, "audio_visual")
print("재계산 F_ext", r["ext"]["F"], "F_basic", r["basic"]["F"], "DOA_ext", r["ext"]["DOA_err"])
old = json.load(open(os.path.join(cfg.run_dir, "scores_val_real.json"), encoding="utf8"))["scores"]
print("Colab   F_ext", old["ext"]["F"], "F_basic", old["basic"]["F"], "DOA_ext", old["ext"]["DOA_err"])
