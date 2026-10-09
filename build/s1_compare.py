# [변경 이력 시작]
#   2026-10-09  최초 생성: S0, S0+S1, S0+S1 9200 결과를 모아 seld_runs/results/s1_comparison.csv 로 저장
# [변경 이력 끝]
"""S0 / S0+S1 / S0+S1(9200회, bs128) 결과를 val(scores_val_real.json)과 test(C:/asrwork/test_*.json, score_dirs 직접 채점)에서 모아
seld_runs/results/s1_comparison.csv 로 저장한다.  python s1_compare.py"""
import json, csv, os
RUNS = r"G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset\seld_runs\runs"
OUT = r"G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset\seld_runs\results\s1_comparison.csv"
COND = [("S0", "_s1", 9200, 64), ("S0+S1", "_s1_S01", 27800, 64), ("S0+S1_9200", "_s1_S01-up9200_bs128", 9246, 128)]
RUNSET = [("main20", p) for p in ("B1", "B2", "B2-0", "B3")] + [("ctrl5", "B3")]
pct = lambda x: "" if x is None else round(x * 100, 2)
rows = []
for ds, p in RUNSET:
    for cond, suf, upd, bs in COND:
        r = f"{ds}_{p}{suf}"
        if not os.path.isdir(os.path.join(RUNS, r)): continue
        row = dict(dataset=ds, preset=p, condition=cond, updates=upd, batch=bs, run=r)
        for split, path in (("val", os.path.join(RUNS, r, "scores_val_real.json")), ("test", rf"C:\asrwork\test_{r}.json")):
            if not os.path.exists(path): continue
            d = json.load(open(path)); s = d.get("scores", d)
            row.update({f"{split}_F_ext": pct(s["ext"]["F"]), f"{split}_F_basic": pct(s["basic"]["F"]),
                        f"{split}_DOA_ext": round(s["ext"]["DOA_err"], 1), f"{split}_onscreen": pct(s["ext"]["onscreen_acc"])})
        for k in ("shuffle", "zeros"):
            path = os.path.join(RUNS, r, f"scores_val_{k}.json")
            if os.path.exists(path): row[f"val_F_ext_{k}"] = pct(json.load(open(path))["scores"]["ext"]["F"])
        rows.append(row)
cols = ["dataset", "preset", "condition", "updates", "batch", "run", "val_F_ext", "val_F_basic", "val_DOA_ext", "val_onscreen", "val_F_ext_shuffle", "val_F_ext_zeros",
        "test_F_ext", "test_F_basic", "test_DOA_ext", "test_onscreen"]
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print(len(rows), "rows ->", OUT)
