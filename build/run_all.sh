#!/bin/bash
# [변경 이력 시작]
#   2026-10-02  최초 생성: 계획이 끝나면 main20, ctrl5의 val, test, train을 차례로 생성하는 실행 스크립트
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
cd "$(dirname "$0")"
while [ ! -f /c/asrwork/dataset/ctrl5/manifest.csv ]; do sleep 10; done
for ds in main20 ctrl5; do for sp in val test train; do
  python build.py gen $ds $sp --workers 10
done; done
echo ALLDONE
