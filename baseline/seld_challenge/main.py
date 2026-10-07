# [변경 이력 시작]
#   2026-10-03  원본의 학습 진입점을 run.py의 train 단계로 넘기는 얇은 껍데기로 교체
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
main.py

원본의 학습 진입점 역할은 이제 run.py 의 `train` 단계가 한다(설정 파일, 프리셋, 이어 하기, 두 방식 채점 포함).
이 파일은 예전 이름으로 실행했을 때 같은 일을 하도록 run.py 로 넘겨 주는 얇은 껍데기다.

    python main.py --dataset main20 --preset B2 --set drive_root=...   ==   python run.py train ...
"""

import sys
import run

if __name__ == '__main__':
    run.main(['train'] + sys.argv[1:])
