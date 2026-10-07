# [변경 이력 시작]
#   2026-10-07  최초 생성: S1 학습 연동 시험용 아주 작은 generated/s1 만들기(gen_v2로 창 6개)
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
import os, sys, shutil, json
os.environ["RAW"] = "C:/asrwork/raw"
sys.path.insert(0, "G:/내 드라이브/Colab Notebooks/ASR_2026-2/for_dataset/build")
import gen_v2 as V, dsgen as G
if __name__ == "__main__":
    out = "C:/asrwork/mini/generated/s1"; shutil.rmtree("C:/asrwork/mini/generated", ignore_errors=True)
    valid = V.load_valid(); split_of, info = G.make_split(valid); vb = {p["name"]: p for p in valid}
    plan, meta = V.make_plan(valid, split_of, 1, 6); plan = plan.head(6)
    V.run_generation(plan, vb, 1, out, "C:/asrwork/mini/genbuf", workers=2, renderer_name="cpu", shard_windows=3, full_verify=True)
    print(V.merge_manifests(out, 1)); print(sorted(os.listdir(out)))
