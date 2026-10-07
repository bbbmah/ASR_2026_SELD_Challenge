# [변경 이력 시작]
#   2026-10-07  최초 생성: 공개 저장소에 올릴 파일을 허용 목록으로 골라 스테이징 폴더에 복사(시드 값 가리기, 노트북 실행 결과 지우기, 비공개 정보 검사)
#   2026-10-07  자기 자신은 토큰 검사에서 제외(검사 문자열 오탐 방지)
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
공개 저장소에 올릴 파일을 골라 정리해서 스테이징 폴더(= git 작업 폴더)에 복사한다. 올리지는 않는다(커밋과 푸시는 따로).

왜 드라이브 폴더를 바로 저장소로 쓰지 않나
 - 공개 저장소라서 **시드 값**을 가려야 한다: 코드와 공개 데이터가 합쳐지면 비공개 test 구성을 재현할 수 있다.
 - 노트북의 코랩 실행 결과(경로, 실행 정보)를 지우고 올려야 한다.
 - 데이터, 체크포인트, test 정답 같은 큰 파일과 비공개 파일이 실수로 들어가지 않게 **허용 목록**으로만 복사한다.
 - .git 을 구글 드라이브 폴더 밖에 둘 수 있다(동기화 충돌 방지).

사용:  python build/publish_sync.py            # 복사하고 검사
       python build/publish_sync.py --dry      # 무엇이 바뀔지만 보기(복사 안 함)
스테이징 폴더: 환경변수 PUBLISH_DST (기본 C:/Users/HwangJunSeo/git/ASR_2026_SELD_Challenge)
"""
import os, re, sys, json, glob

SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DST = os.environ.get("PUBLISH_DST", "C:/Users/HwangJunSeo/git/ASR_2026_SELD_Challenge")
DRY = "--dry" in sys.argv

# 올릴 파일(허용 목록, SRC 기준 glob). 여기 없는 것은 올라가지 않는다.
INCLUDE = ["*.md", "*.ipynb",
           "build/*.py", "build/*.md", "build/*.ps1", "build/*.sh", "build/*.yaml", "build/*.yml", "build/*.ipynb", "build/*.txt",
           "baseline/seld_challenge/*.py", "baseline/seld_challenge/*.md", "baseline/seld_challenge/*.ipynb", "baseline/seld_challenge/configs/*",
           "dataset/*.md",
           "seld_runs/results/*.csv", "seld_runs/results/*.json", "seld_runs/results/*.md",
           "seld_runs/runs/*/train_log.csv", "seld_runs/runs/*/scores_*.json", "seld_runs/runs/*/train_summary.json", "seld_runs/runs/*/config_used.yaml"]
FORBIDDEN_NAMES = re.compile(r"(^|/)(private|manifest[^/]*\.csv|plan\.csv|probe_valid\.json)(/|$)")        # 비공개 정보가 든 이름
FORBIDDEN_EXT = (".pt", ".pth", ".zip", ".npz", ".mp4", ".wav", ".pkl")
MAX_BYTES = 2_000_000
GITIGNORE = "__pycache__/\n*.pyc\n*.pt\n*.pth\n*.zip\n*.npz\n*.mp4\n*.wav\n*.pkl\nprivate/\nmanifest*.csv\n"      # 2중 안전장치


def find_seed():
    """비공개 시드 값(빌드 스크립트의 SEED). 이 스크립트에는 값을 적어 두지 않는다."""
    t = open(os.path.join(SRC, "build", "build.py"), encoding="utf-8").read()
    m = re.search(r"(?m)^SEED\s*=\s*(\d+)", t)
    return m.group(1) if m else None


SEED = find_seed()


def redact(text, ext):
    """시드 값을 가린다. 코드는 0(문법 유지), 문서는 <비공개>."""
    if not SEED:
        return text
    return text.replace(SEED, "0" if ext in (".py", ".ps1", ".sh", ".yaml", ".yml") else "<비공개>")


def sanitize(rel, raw):
    ext = os.path.splitext(rel)[1].lower()
    if ext == ".ipynb":
        nb = json.loads(raw.decode("utf-8"))
        for c in nb.get("cells", []):
            if c.get("cell_type") == "code":
                c["outputs"] = []; c["execution_count"] = None
            c.pop("id", None)
            c["metadata"] = {}
            src = "".join(c["source"]); c["source"] = redact(src, ".py" if c["cell_type"] == "code" else ".md").splitlines(True)
        md = nb.get("metadata", {})
        nb["metadata"] = {k: md[k] for k in ("kernelspec", "language_info", "accelerator") if k in md}      # 코랩 실행 정보(colab, widgets 등)는 뺀다
        return (json.dumps(nb, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    if ext in (".py", ".md", ".ps1", ".sh", ".yaml", ".yml", ".txt", ".csv", ".json"):
        t = raw.decode("utf-8-sig") if raw.startswith(b"\xef\xbb\xbf") else raw.decode("utf-8")
        t = redact(t, ext if ext in (".py", ".ps1", ".sh", ".yaml", ".yml") else ".md")
        return t.replace("\r\n", "\n").encode("utf-8")
    return raw


def collect():
    files = {}
    for pat in INCLUDE:
        for p in glob.glob(os.path.join(SRC, pat)):
            if not os.path.isfile(p):
                continue
            rel = os.path.relpath(p, SRC).replace("\\", "/")
            if rel.lower().endswith(FORBIDDEN_EXT) or FORBIDDEN_NAMES.search(rel):
                continue
            files[rel] = p
    return files


def main():
    files = collect(); out = {}
    for rel, p in sorted(files.items()):
        raw = open(p, "rb").read()
        if len(raw) > MAX_BYTES:
            sys.exit(f"[중단] 너무 큰 파일: {rel} ({len(raw)}바이트)")
        out[rel] = sanitize(rel, raw)
    out[".gitignore"] = GITIGNORE.encode("utf-8")
    # 검사: 가린 뒤에도 시드, 이메일, 토큰이 남아 있으면 중단
    bad = []
    for rel, b in out.items():
        t = b.decode("utf-8", "replace")
        if SEED and SEED in t: bad.append((rel, "시드"))
        if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", t): bad.append((rel, "이메일"))
        if rel != "build/publish_sync.py" and re.search(r"(ghp_|github_pat_|AKIA[0-9A-Z]{8}|BEGIN (RSA |OPENSSH )?PRIVATE KEY)", t):     # 이 스크립트는 검사 문자열 때문에 제외
            bad.append((rel, "토큰류"))
    if bad:
        sys.exit(f"[중단] 올리면 안 되는 내용이 남아 있음: {bad}")
    # 비교와 복사
    added, changed, same = [], [], 0
    for rel, b in out.items():
        dst = os.path.join(DST, rel)
        if not os.path.exists(dst): added.append(rel)
        elif open(dst, "rb").read() != b: changed.append(rel)
        else: same += 1
    removed = []
    if os.path.isdir(DST):
        for dp, dn, fn in os.walk(DST):
            dn[:] = [d for d in dn if d != ".git"]
            for f in fn:
                rel = os.path.relpath(os.path.join(dp, f), DST).replace("\\", "/")
                if rel not in out: removed.append(rel)
    print(f"올릴 파일 {len(out)}개 ({sum(len(b) for b in out.values()) / 1e6:.2f}MB) | 새 파일 {len(added)}, 바뀜 {len(changed)}, 같음 {same}, 지워짐 {len(removed)} | 시드 가림: {'예' if SEED else '시드 못 찾음'}")
    for lab, lst in (("새", added), ("바뀜", changed), ("삭제", removed)):
        for r in lst[:60]: print(f"  [{lab}] {r}")
    if DRY:
        print("(--dry: 복사하지 않음)"); return
    for rel, b in out.items():
        if rel in added or rel in changed:
            dst = os.path.join(DST, rel); os.makedirs(os.path.dirname(dst), exist_ok=True); open(dst, "wb").write(b)
    for rel in removed:
        os.remove(os.path.join(DST, rel))
    print("복사 완료:", DST)


if __name__ == "__main__":
    main()
