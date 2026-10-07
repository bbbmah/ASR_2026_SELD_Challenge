# [변경 이력 시작]
#   2026-10-06  최초 생성: 같은 20초 창 단위 부트스트랩으로 점수, 격차, main20-ctrl5 차이의 95% 구간 계산
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""eval_all.py 가 저장한 클립별 집계값으로 점수, 격차, 창 단위 부트스트랩 95% 구간을 계산한다.
- 단위: main20 의 20초 창. ctrl5 는 같은 창의 조각 4개를 합쳐서 한 단위로 센다(parent_map).
- 모든 모델에 같은 창을 뽑는다(쌍 부트스트랩). 불확실성은 '평가 창을 다시 뽑을 때의 흔들림'이며, 학습 시드 간 흔들림은 포함하지 않는다.
python analyze_results.py"""
import os, json, csv
import numpy as np

EV = r"C:\asrwork\evalall"
PM = r"C:\asrwork\dataset\ctrl5\private\parent_map.csv"
PRESETS = ["B1", "B2", "B2-0", "B3"]
DSS = ["main20", "ctrl5"]
B = 2000
EPS = np.finfo(float).eps
I = dict(TP=0, FP=1, FPs=2, FN=3, Nref=4, Ang=5, Rel=6, Dist=7, Onsc=8, DETP=9)

parent = {}
with open(PM, newline="", encoding="utf8") as f:
    for r in csv.DictReader(f): parent[r["clip"]] = r["parent_clip"]


def load(ds, p, split):
    z = np.load(os.path.join(EV, f"{ds}_{p}_s1_{split}.npz"))
    names, arr = list(z["names"]), z["arr"]                     # arr: (2, n, 10, 13)
    win = [parent[n] if ds == "ctrl5" else n for n in names]
    return win, arr


def to_windows(ds, p, split, order):
    win, arr = load(ds, p, split)
    idx = {w: i for i, w in enumerate(order)}
    out = np.zeros((2, len(order), 10, 13))
    for k, w in enumerate(win): out[:, idx[w]] += arr[:, k]
    return out


def metrics(S):
    """S: (2, 10, 13) 합계 -> dict. 0=ext(접지 않음), 1=basic(접음)"""
    res = {}
    for f, tag in ((0, "ext"), (1, "basic")):
        s = S[f]
        TP, FP, FPs, FN, Nref = s[I["TP"]], s[I["FP"]], s[I["FPs"]], s[I["FN"]], s[I["Nref"]]
        F = TP / (EPS + TP + FPs + 0.5 * (FP + FN)); F = np.where(Nref == 0, np.nan, F)
        de = s[I["DETP"]]
        def cm(key):
            v = s[I[key]] / (de + EPS); v = np.where(de == 0, np.nan, v); return np.nanmean(v) if (de > 0).any() else np.nan
        res[f"F_{tag}"] = np.nanmean(F) if (~np.isnan(F)).any() else np.nan
        res[f"DOA_{tag}"] = cm("Ang"); res[f"onscreen_{tag}"] = cm("Onsc"); res[f"relDist_{tag}"] = cm("Rel")
        res[f"Fclass_{tag}"] = F
    res["frontback"] = res["DOA_ext"] - res["DOA_basic"]
    return res


def main():
    out = {}
    for split in ("val", "test"):
        order = sorted(set(load("main20", "B1", split)[0]))
        assert set(load("ctrl5", "B1", split)[0]) == set(order), "창 목록이 다릅니다"
        W = {(ds, p): to_windows(ds, p, split, order) for ds in DSS for p in PRESETS}
        n = len(order)
        point = {k: metrics(W[k].sum(1)) for k in W}
        rng = np.random.default_rng(20261006)
        boot = {k: {m: np.zeros(B) for m in point[k] if not m.startswith("Fclass")} for k in W}
        for b in range(B):
            idx = rng.integers(0, n, n)
            for k in W:
                r = metrics(W[k][:, idx].sum(1))
                for m in boot[k]: boot[k][m][b] = r[m]
        def ci(x): return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]
        def pval(x):
            x = x[~np.isnan(x)]; return float(2 * min((x > 0).mean(), (x < 0).mean()))
        S = dict(n_windows=n, models={}, gaps={}, did={})
        for (ds, p) in W:
            S["models"][f"{ds}_{p}"] = {m: dict(v=float(point[(ds, p)][m]), ci=ci(boot[(ds, p)][m]))
                                         for m in boot[(ds, p)]}
            S["models"][f"{ds}_{p}"]["Fclass_ext"] = [None if np.isnan(v) else float(v) for v in point[(ds, p)]["Fclass_ext"]]
            S["models"][f"{ds}_{p}"]["Fclass_basic"] = [None if np.isnan(v) else float(v) for v in point[(ds, p)]["Fclass_basic"]]
        gaps = [("B2-B1", "B2", "B1"), ("B2-(B2-0)", "B2", "B2-0"), ("B2-0 - B1", "B2-0", "B1"), ("B3-B1", "B3", "B1"), ("B3-B2", "B3", "B2"), ("B3-(B2-0)", "B3", "B2-0")]
        for m in ("F_ext", "F_basic", "DOA_ext", "DOA_basic", "frontback", "onscreen_ext"):
            for name, a, c in gaps:
                if m == "onscreen_ext" and "B1" in (a, c): continue
                d = {}
                for ds in DSS:
                    x = boot[(ds, a)][m] - boot[(ds, c)][m]
                    d[ds] = dict(v=float(point[(ds, a)][m] - point[(ds, c)][m]), ci=ci(x), p=pval(x))
                x = (boot[("main20", a)][m] - boot[("main20", c)][m]) - (boot[("ctrl5", a)][m] - boot[("ctrl5", c)][m])
                d["main20-ctrl5"] = dict(v=d["main20"]["v"] - d["ctrl5"]["v"], ci=ci(x), p=pval(x))
                S["gaps"][f"{m}|{name}"] = d
        out[split] = S
    json.dump(out, open(r"C:\asrwork\analysis.json", "w"), indent=1)
    # 검증: 점수 파일과 같은 값인지 (val)
    base = r"G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset\seld_runs\runs"
    mx = 0
    for ds in DSS:
        for p in PRESETS:
            s = json.load(open(f"{base}/{ds}_{p}_s1/scores_val_real.json", encoding="utf8"))["scores"]
            mine = out["val"]["models"][f"{ds}_{p}"]
            for k, mk in (("F", "F_ext"),):
                mx = max(mx, abs(s["ext"][k] - mine[mk]["v"]))
            mx = max(mx, abs(s["basic"]["F"] - mine["F_basic"]["v"]), abs(s["ext"]["DOA_err"] - mine["DOA_ext"]["v"]))
    print("val 점수 파일과 최대 차이:", mx)
    for ds in DSS:
        for p in PRESETS:
            fn = rf"C:\asrwork\main20_{p}_s1_test_scores.json"
            if ds == "main20" and os.path.exists(fn):
                t = json.load(open(fn)); mine = out["test"]["models"][f"main20_{p}"]
                print("main20 test", p, "score.py F_ext", round(t["ext"]["F"], 5), "내 계산", round(mine["F_ext"]["v"], 5), "| DOA", round(t["ext"]["DOA_err"], 3), round(mine["DOA_ext"]["v"], 3))


if __name__ == "__main__":
    main()
