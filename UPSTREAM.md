<!-- [변경 이력 시작]
  2026-10-07  최초 생성: 사용한 남의 코드(DCASE2025 베이스라인, Sony 데이터 생성기)의 주소와 커밋
  2026-10-07  변경 이력 주석 추가
[변경 이력 끝] -->

# 사용한 남의 코드 (저장소에는 넣지 않음)

| 저장소 | 쓴 커밋 | 쓰임 |
|---|---|---|
| https://github.com/partha2409/DCASE2025_seld_baseline | `42a48b6456b73be35ad0e1a9ffeb6ceef83ae0bd` (2025-05-27, "added support for inference on eval data") | 베이스라인 모델 코드. `baseline/seld_challenge/`는 이 커밋의 작업 폴더를 복사해 고친 것입니다(고친 곳은 `CHANGES.md`). 원본은 고치지 않았습니다 |
| https://github.com/SonyResearch/dcase2025_stereo_seld_data_generator | `08e267e5152d37cb7a828f5f52b256f917609132` (2025-07-17, "Update README.md") | 스테레오 변환과 영상 생성의 참고 코드. 필요한 부분만 가져다 `build/dsgen.py`에 다시 썼고, 비교 시험(렌더링, onscreen)에만 직접 호출했습니다. 이 저장소에는 `LICENSE`가 있습니다 |

두 저장소 모두 같은 커밋을 다시 받아(`git clone` 후 `git checkout <커밋>`) 확인할 수 있습니다. DCASE2025 베이스라인 저장소에는 라이선스 파일이 없어서, 고친 복사본(`baseline/seld_challenge/`)의 재배포 조건은 원 저자를 따르세요.

데이터는 STARSS23(https://zenodo.org/ 에서 배포)을 썼고, 이 저장소에 포함하지 않았습니다.
