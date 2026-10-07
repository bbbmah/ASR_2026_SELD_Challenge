# [변경 이력 시작]
#   2026-10-03  최초 생성: 예측 폴더를 정답 폴더로 채점하는 스크립트(두 방식 점수, 제외 목록)
#   2026-10-03  onscreen 정확도가 없을 때 'NA%'로 표시되던 것을 'NA'로 수정
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
score.py

예측 csv 폴더를 정답 csv 폴더로 채점한다. 학습 중 val 채점, 최종 val 채점, test 채점(주최자용), 리더보드 채점이 모두 이 파일을 쓴다.
torch가 필요 없다.

항상 두 방식으로 계산한다.
  ext   확장 점수: 방위각을 접지 않는다. 순위와 모델 고르기에 쓴다.
  basic 기본 점수: 정답과 예측을 둘 다 앞쪽으로 접는다(DCASE 방식).
소리만 쓰는 모델(--modality audio)은 onscreen을 내지 않으므로 onscreen 정확도는 NA(null)다.
"영상 덕분에 오른 점수"는 onscreen을 요구하지 않는 F(lad_req_onscreen=False)로만 계산한다.

사용 예:
  python score.py --pred runs/main20_B2_s1/pred_val --ref /content/data/main20/val/labels --clip_len 200 \
      --modality audio_visual --out scores.json
  python score.py ... --exclude_list configs/exclude_val_fix20.txt            # 클립 이름 한 줄에 하나
  python score.py ... --exclude_list configs/exclude_val_fix20.txt \
      --parent_map /content/data/ctrl5/private/parent_map.csv                  # ctrl5: 부모 main20 클립 이름을 조각 이름으로 바꿔서 뺀다
"""

import os
import csv
import json
import argparse
from metrics import ComputeSELDResults

DEFAULT_SCORE_PARAMS = dict(lad_doa_thresh=20, lad_dist_thresh=float('inf'), lad_reldist_thresh=1.0,
                            lad_req_onscreen=False, average='macro', nb_classes=13, exclude_absent_classes=True)


def read_exclude_list(path, parent_map=None):
    """제외할 클립 이름 집합. parent_map이 있으면 목록의 이름을 부모(main20) 클립으로 보고 그 조각들로 바꾼다."""
    names = set()
    if path:
        with open(path, encoding='utf8') as f:
            for line in f:
                line = line.split('#')[0].strip()
                if line:
                    names.add(line[:-4] if line.endswith('.csv') else line)
    if parent_map and names:
        pieces = set()
        with open(parent_map, newline='', encoding='utf8') as f:
            for row in csv.DictReader(f):
                if row['parent_clip'] in names:
                    pieces.add(row['clip'])
        names = pieces
    return names


def score_dirs(pred_dir, ref_dir, clip_len, modality='audio_visual', exclude=None, exclude_absent_classes=True, **overrides):
    params = dict(DEFAULT_SCORE_PARAMS, modality=modality, exclude_absent_classes=exclude_absent_classes, **overrides)
    res = ComputeSELDResults(params, ref_files_folder=ref_dir, clip_len=clip_len, exclude=exclude)
    return res.get_SELD_Results(pred_dir)


def one_line(res):
    pct = lambda v: 'NA' if v is None else f"{v * 100:.1f}%"
    num = lambda v: 'NA' if v is None else f"{v:.2f}"
    parts = []
    for k in ('ext', 'basic'):
        r = res[k]
        parts.append(f"[{k}] F={pct(r['F'])} DOA={num(r['DOA_err'])} RelDist={num(r['rel_dist_err'])} Onscreen={pct(r['onscreen_acc'])}")
    return f"clips={res['n_clips']} " + " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='예측 csv 폴더')
    ap.add_argument('--ref', required=True, help='정답 csv 폴더')
    ap.add_argument('--clip_len', type=int, required=True, help='라벨 프레임 수 (main20/fix20 200, ctrl5 50)')
    ap.add_argument('--modality', default='audio_visual', choices=['audio', 'audio_visual'])
    ap.add_argument('--exclude_list', default=None, help='뺄 클립 이름 목록 파일')
    ap.add_argument('--parent_map', default=None, help='ctrl5/private/parent_map.csv (목록을 부모 main20 클립 이름으로 볼 때)')
    ap.add_argument('--keep_absent_classes', action='store_true', help='정답이 없는 클래스도 평균 F에 넣는다')
    ap.add_argument('--out', default=None, help='json 저장 경로')
    a = ap.parse_args()
    res = score_dirs(a.pred, a.ref, a.clip_len, a.modality, read_exclude_list(a.exclude_list, a.parent_map),
                     exclude_absent_classes=not a.keep_absent_classes)
    print(one_line(res))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
