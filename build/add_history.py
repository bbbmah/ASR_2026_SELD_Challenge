# [변경 이력 시작]
#   2026-10-07  최초 생성: 파일마다 변경 이력 주석을 넣는 스크립트(이력 데이터 HIST가 이 파일에 있음)
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
파일마다 "변경 이력"(최초 생성일부터 수정일과 간단한 내용)을 주석으로 넣거나 갱신한다.
 - 이력 데이터는 아래 HIST 에만 있다. 새로 고친 파일이 있으면 HIST 에 한 줄을 추가하고 `python build/add_history.py` 를 다시 실행한다.
 - 같은 블록(시작/끝 표시 사이)을 교체하므로 여러 번 실행해도 중복되지 않는다.
 - 형식: py, ps1, sh, yaml, txt = '#' 주석 / md = 맨 위 HTML 주석 / ipynb = 첫 마크다운 셀 맨 위 HTML 주석.
 - 날짜는 파일의 생성 시각, 수정 시각과 작업 기록(대화 기록)을 맞춰 적었다. 복사로 만든 파일은 복사한 날짜를 적고 그렇게 표시했다.
 - 사용자가 직접 쓴 지시 파일(CLAUDE.md, MODEL_CODE_CH.md, rev_Ctrl5.md)과 데이터 성격의 파일(mini_common.txt, pip_freeze_*.txt, conda_env_*.yml)은 건드리지 않는다.
"""
import os, sys, json, re

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
START, END = "[변경 이력 시작]", "[변경 이력 끝]"
TODAY = "2026-10-07"
LAST = (TODAY, "변경 이력 주석 추가")

HIST = {
    # ------------------------------------------------------------------ 루트 문서
    "CHANGES.md": [
        ("2026-10-03", "최초 생성: 원본 베이스라인 대비 변경 목록(필수 수정 10개, 추가로 찾은 것, 새 모델 옵션, 정한 기본값)"),
        ("2026-10-03", "소규모 시험 결과 반영: 제외 목록 평가를 별도 파일(_excl)로 저장, 학습 클립이 배치보다 적을 때 에러 항목 추가"),
        ("2026-10-07", "6절 추가: 추가 학습 데이터(S1) 연동 변경(train_stages, updates, 필터, datainfo, data_gaps)"),
    ],
    "CHECK_REPORT.md": [
        ("2026-10-03", "최초 생성: 필수 확인 결과(7.1 소규모 학습은 보류로 기록, 7.2 채점 시험 3개, 7.3 코드 읽기 확인 2개)"),
        ("2026-10-03", "7.1을 conda 환경(CPU)에서 실제로 돌린 소규모 학습 시험 결과로 교체, 시험 중 찾아 고친 문제 3가지 기록"),
    ],
    "README_baseline.md": [
        ("2026-10-03", "최초 생성: 코랩 사용 설명서(순서, 설정의 뜻, 프리셋, 결과 읽는 법, 문제 해결)"),
        ("2026-10-03", "소규모 시험 결과 반영: 주의 문구 수정"),
        ("2026-10-07", "9절 추가(추가 학습 데이터 S1로 학습하기), 학습 시간 실측값 반영, 주의 문구와 준비물 폴더(generated/) 갱신"),
    ],
    "README_generate.md": [
        ("2026-10-07", "최초 생성: 데이터 생성 노트북(colab_generate.ipynb) 사용 설명서"),
        ("2026-10-07", "학습 쪽 연동이 구현되었음을 반영(S1 생성 완료)"),
    ],
    "RESULTS_REPORT.md": [
        ("2026-10-06", "최초 생성: main20, ctrl5 x B1, B2, B2-0, B3 학습 결과 보고(점수, 격차, 창 단위 부트스트랩 구간, 클래스별, 한계)"),
        ("2026-10-06", "10절 추가: DCASE 2025 점수와의 관계, onscreen을 요구한 F 재채점. ctrl5 B2 F_ext 표 값 정정(9.17 -> 9.20)"),
        ("2026-10-09", "11절 추가: S0 대 S0+S1(27,800회, 9,200회 bs128) 비교. main20 4종과 ctrl5 B3의 val, test 점수, 영상 효과, 한계"),
    ],
    "build/s1_compare.py": [
        ("2026-10-09", "최초 생성: S0, S0+S1, S0+S1 9200 결과를 모아 seld_runs/results/s1_comparison.csv 로 저장"),
    ],
    "DATA_PLAN_v2.md": [
        ("2026-10-06", "최초 생성: 데이터 확장(시간 창, 카메라 변형, 좌우 거울)과 학습 시 증강 플랜"),
        ("2026-10-07", "저장 제약 정정: 드라이브는 5TB이고 G: 여유는 스트리밍 캐시 값. 로컬은 생성 버퍼로 쓰는 방식으로 변경, 로컬 정리의 영향 추가"),
        ("2026-10-07", "구현 상태 추가: 생성 코드(gen_v2.py)와 코랩 노트북 구현"),
        ("2026-10-07", "S1 생성 완료와 학습 쪽 연동 구현을 반영"),
    ],
    "LOCAL_CLEANUP_PLAN.md": [
        ("2026-10-06", "최초 생성: 로컬(C:/asrwork) 정리 계획과 드라이브 사본 점검 결과(파일 대조, zip 검사)"),
        ("2026-10-07", "실행 결과(8절) 추가(A, B, C 단계 삭제), 드라이브 여유 29GB 전제를 정정, 경로 제어문자 오류 수정"),
    ],
    # ------------------------------------------------------------------ dataset/ 폴더 문서
    "dataset/INTRO_dataset.md": [
        ("2026-10-02", "최초 생성: 챌린지와 데이터셋을 처음 접하는 사람을 위한 소개글. 이후 변경 없음(main20 숫자는 최초 생성 당시 값)"),
    ],
    "dataset/BUILD_REPORT.md": [
        ("2026-10-02", "최초 생성: main20, ctrl5(v1) 생성 결과 보고(split별 통계, 챌린지 성립 지표, 검증 결과)"),
        ("2026-10-03", "ctrl5 v2와 fix20 작업 반영: 보고서 재조립(ctrl5 v2 절 추가, 기존 ctrl5는 v1 폐기로 표기, main20 숫자 갱신)"),
    ],
    "dataset/BUILD_REPORT.v1.md": [
        ("2026-10-03", "생성: report_v2.py가 재조립하기 전의 보고서 원문을 백업한 파일(수정 없음)"),
    ],
    "dataset/NOTES.md": [
        ("2026-10-02", "최초 생성(build/NOTES.md의 사본): 정한 것과 기본값 기록"),
        ("2026-10-03", "ctrl5 v2/fix20 작업 절 추가: 빈 라벨 처리와 채점 단위 확인 결과, 피크로 빠진 창, fix20 결과"),
    ],
    "dataset/README_dataset.md": [
        ("2026-10-02", "최초 생성(build/README_dataset.md의 사본): 데이터셋 형식 설명(폴더, 오디오, 영상, 라벨, 좌표 규약, 분할)"),
        ("2026-10-03", "ctrl5 v2 설명 절 추가(main20 창을 5초씩 4등분, 내부용, 라벨 없는 조각 있음)"),
    ],
    "seld_runs/results/RESULTS_REPORT.md": [
        ("2026-10-06", "상위 폴더 RESULTS_REPORT.md의 사본을 결과 폴더에 둠. 본문 변경 이력은 원본과 같다(최초 생성, 10절 추가, 표 값 정정)"),
        ("2026-10-09", "상위 RESULTS_REPORT.md의 11절(S0 대 S0+S1 비교) 반영한 사본으로 교체"),
    ],
    # ------------------------------------------------------------------ build/ 문서
    "build/NOTES.md": [
        ("2026-10-02", "최초 생성: 데이터셋 생성에서 정한 것과 기본값"),
        ("2026-10-03", "ctrl5 v2/fix20 작업 절 추가: 확인 결과, 피크로 빠진 창, 코드 실수 기록"),
    ],
    "build/README_dataset.md": [
        ("2026-10-02", "최초 생성: 데이터셋 형식 설명"),
        ("2026-10-03", "ctrl5 v2 설명 절 추가"),
    ],
    "build/report_tail.md": [
        ("2026-10-02", "최초 생성: BUILD_REPORT.md의 꼬리 부분(파일럿 속도, 검증 결과, 틀렸던 가정). 이후 변경 없음"),
    ],
    # ------------------------------------------------------------------ build/ 코드 (데이터셋 생성)
    "build/dsgen.py": [
        ("2026-10-02", "최초 생성: 카메라 궤적, 영상 렌더러(투시 변환), FOA에서 스테레오 오디오 회전, 라벨 변환, 분할(split). 이후 변경 없음(다른 스크립트가 그대로 가져다 씀)"),
    ],
    "build/validate.py": [
        ("2026-10-02", "최초 생성: 필수 검증 5개(입력 확인, 좌표 단위 시험, 렌더링 비교, 오디오 회전 시험, onscreen 비교)"),
        ("2026-10-02", "그림 저장을 cv2.imencode로 변경(한글 경로에서 imwrite가 조용히 실패함)"),
    ],
    "build/build.py": [
        ("2026-10-02", "최초 생성: 클립 계획(plan)과 병렬 생성(gen), main20과 ctrl5(v1). 이어 하기, 실패 로그 포함"),
    ],
    "build/finalize.py": [
        ("2026-10-02", "최초 생성: 통계, 챌린지 성립 지표, BUILD_REPORT 작성, 공개 zip 포장"),
        ("2026-10-03", "공개 묶음 만들 때 피크로 빠진(dropped) main20 클립을 제외하도록 수정"),
    ],
    "build/run_all.sh": [
        ("2026-10-02", "최초 생성: 계획이 끝나면 main20, ctrl5의 val, test, train을 차례로 생성하는 실행 스크립트"),
    ],
    "build/build_v2.py": [
        ("2026-10-03", "최초 생성: ctrl5 v2(main20 창을 5초 4조각, 고정 카메라)와 fix20 생성, 피크 탈락 반영, 이름 매기기"),
        ("2026-10-03", "pandas 메서드와 겹치는 m.clip을 m[\"clip\"]으로 수정, fix20 파일 이동 코드 단순화"),
    ],
    "build/validate_ctrl5.py": [
        ("2026-10-03", "최초 생성: ctrl5 v2 필수 확인 5개(개수, 같은 구간, 빈 라벨 읽기, 채점 단위, 공개 묶음 점검)"),
        ("2026-10-03", "m.clip 오류를 [\"clip\"]으로 수정"),
    ],
    "build/report_v2.py": [
        ("2026-10-03", "최초 생성: BUILD_REPORT.md 재조립(ctrl5 v2 절 추가, v1 폐기 표기, fix20 절)"),
    ],
    "build/make_internal_zips.py": [
        ("2026-10-03", "최초 생성: ctrl5, fix20 내부 전송용 zip(internal_*) 만들기"),
        ("2026-10-07", "설명 문자열의 윈도우 경로를 슬래시로 바꿔 파이썬 SyntaxWarning('\\i') 제거(동작 변경 없음)"),
    ],
    "build/make_mini.py": [
        ("2026-10-03", "최초 생성: 소규모 시험용 미니 데이터와 zip 만들기(데이터셋마다 train 4, val 2, test 2)"),
    ],
    "build/mini_all.ps1": [
        ("2026-10-03", "최초 생성(복사한 날짜 기준): 미니 데이터로 2 데이터셋 x 프리셋 4개의 전 단계를 돌리는 스크립트"),
    ],
    "build/mini_gaps.ps1": [
        ("2026-10-03", "최초 생성(복사한 날짜 기준): 접미사 없는 실행 이름으로 gaps 계산, fix20 평가, 제외 목록 시험"),
    ],
    "build/mini_base.yaml": [
        ("2026-10-03", "최초 생성(복사한 날짜 기준): 소규모 시험용 설정(epochs 1, batch_size 2, num_workers 0)"),
        ("2026-10-07", "새 base.yaml(추가 학습 데이터 항목 포함)에서 다시 만듦"),
    ],
    # ------------------------------------------------------------------ build/ 코드 (평가, 분석, 정리)
    "build/eval_all.py": [
        ("2026-10-06", "최초 생성: 8개 모델의 val, test 예측을 클립별 집계값으로 채점(부트스트랩 분석의 입력)"),
        ("2026-10-06", "드라이브 사본이 불완전한 ctrl5_B2 val 예측을 로컬 재생성본으로 대체, 일부 작업만 다시 돌리는 인자 추가"),
    ],
    "build/analyze_results.py": [
        ("2026-10-06", "최초 생성: 같은 20초 창 단위 부트스트랩으로 점수, 격차, main20-ctrl5 차이의 95% 구간 계산"),
    ],
    "build/recompute_pred.py": [
        ("2026-10-06", "최초 생성: 드라이브 사본이 불완전한 ctrl5_B2 val 예측을 저장된 최고 모델로 로컬에서 다시 만들고 Colab 점수와 대조"),
    ],
    "build/eval_onscreen.py": [
        ("2026-10-06", "최초 생성: 영상+소리 모델을 onscreen까지 맞아야 정답인 조건(공식 F(20/1/on))으로 다시 채점"),
    ],
    "build/verify_sync.py": [
        ("2026-10-06", "최초 생성: 로컬과 드라이브 사본의 파일 목록, 크기, zip 목록과 CRC를 읽기 전용으로 대조"),
    ],
    "build/cleanup_local.py": [
        ("2026-10-07", "최초 생성: 로컬 정리 A, B, C. 기본은 dry-run, --execute일 때만 삭제하고 삭제 전 드라이브 사본 크기 대조, 삭제 목록 저장"),
    ],
    # ------------------------------------------------------------------ build/ 코드 (추가 데이터 생성과 학습 연동)
    "build/gen_v2.py": [
        ("2026-10-07", "최초 생성: 추가 데이터 생성 모듈(균일 격자 창 계획, 카메라 가족 F1~F5, 좌우 거울, 창 하나에서 main20 클립과 ctrl5 조각 생성, GPU 렌더러 옵션, 묶음 생성/zip/업로드 검증/이어 하기, 자체 점검)"),
        ("2026-10-07", "영상 임시 파일 이름을 .tmp.mp4로(OpenCV 컨테이너), JSON 출력 정리, 작은 목표에서 격자 간격 발산 방지, 매니페스트 병합 함수 추가"),
    ],
    "build/make_generate_notebook.py": [
        ("2026-10-07", "최초 생성: colab_generate.ipynb를 만드는 스크립트"),
        ("2026-10-07", "GPU 렌더러일 때 워커 수를 최대 6개로 제한(워커마다 약 2GB 메모리)"),
    ],
    "build/nb_harness.py": [
        ("2026-10-07", "최초 생성: 노트북 셀 코드를 코랩 없이(드라이브 연결, 원천 풀기 제외) 실행해 보는 하네스"),
    ],
    "build/make_train_notebook.py": [
        ("2026-10-07", "최초 생성: 학습 노트북 colab_run.ipynb를 만드는 스크립트(추가 학습 데이터 S1 설정 칸, datainfo 셀, data_gaps 출력, 추천 실험 순서)"),
    ],
    "build/make_tiny_s1.py": [
        ("2026-10-07", "최초 생성: S1 학습 연동 시험용 아주 작은 generated/s1 만들기(gen_v2로 창 6개)"),
    ],
    "build/real_s1_smoke.py": [
        ("2026-10-07", "최초 생성: 드라이브의 실제 S1 zip 일부로 이름 규칙, 특징 추출, 데이터 로더 모양을 확인"),
    ],
    "build/run_s1_mini.ps1": [
        ("2026-10-07", "최초 생성: 미니 S1 학습 연동 시험(prepare, features, datainfo, train, eval, swaptest, summary) 실행 스크립트"),
    ],
    "README.md": [
        ("2026-10-07", "최초 생성: 공개 GitHub 저장소용 개요(저장소에 없는 것, 문서 지도, 폴더 설명, 변경 이력 규칙)"),
    ],
    "UPSTREAM.md": [
        ("2026-10-07", "최초 생성: 사용한 남의 코드(DCASE2025 베이스라인, Sony 데이터 생성기)의 주소와 커밋"),
    ],
    "build/publish_sync.py": [
        ("2026-10-07", "최초 생성: 공개 저장소에 올릴 파일을 허용 목록으로 골라 스테이징 폴더에 복사(시드 값 가리기, 노트북 실행 결과 지우기, 비공개 정보 검사)"),
        ("2026-10-07", "자기 자신은 토큰 검사에서 제외(검사 문자열 오탐 방지)"),
    ],
    "build/add_history.py": [
        ("2026-10-07", "최초 생성: 파일마다 변경 이력 주석을 넣는 스크립트(이력 데이터 HIST가 이 파일에 있음)"),
    ],
    # ------------------------------------------------------------------ baseline/seld_challenge (원본 DCASE2025_seld_baseline 복사본을 고친 것)
    "baseline/seld_challenge/README.md": [
        ("2026-10-03", "원본 baseline/DCASE2025_seld_baseline/README.md를 복사. 내용은 고치지 않음(지금 코드와 맞지 않는 부분이 있어서 사용법은 README_baseline.md를 볼 것)"),
    ],
    "baseline/seld_challenge/parameters.py": [
        ("2026-10-03", "원본에서 복사 후 수정: 데이터 경로, fold 이름, 라벨 길이, 배치 크기 등을 configs/base.yaml과 config.py로 옮기고 새 옵션 키(video_input, video_crop, av_time_window 등) 추가"),
    ],
    "baseline/seld_challenge/seld_label_utils.py": [
        ("2026-10-03", "최초 생성: 원본 utils.py에서 torch 없이 쓰는 라벨 읽기와 각도 함수를 분리, 접지 않는 원형 각도 차이 추가"),
    ],
    "baseline/seld_challenge/metrics.py": [
        ("2026-10-03", "원본에서 복사 후 수정: 확장(접지 않음)과 기본(접음) 점수를 한 번에 계산, 채점 길이를 클립 길이로, 정답 폴더를 하위 폴더 없이 읽기, 정답 없는 클래스 평균 제외 옵션, 예측 csv 없는 클립은 빈 예측으로 채점"),
    ],
    "baseline/seld_challenge/score.py": [
        ("2026-10-03", "최초 생성: 예측 폴더를 정답 폴더로 채점하는 스크립트(두 방식 점수, 제외 목록)"),
        ("2026-10-03", "onscreen 정확도가 없을 때 'NA%'로 표시되던 것을 'NA'로 수정"),
    ],
    "baseline/seld_challenge/utils.py": [
        ("2026-10-03", "원본에서 복사 후 수정: load_video가 30fps를 가정하던 것을 실제 fps로 계산하고 프레임 수 검사 추가, 라벨 함수는 seld_label_utils로 분리, onscreen 확률을 int()로 잘라 저장하던 버그를 0.5 기준으로 수정, setup()과 print_results() 제거"),
    ],
    "baseline/seld_challenge/model.py": [
        ("2026-10-03", "원본에서 복사 후 수정: video_input(zeros) 옵션, av_time_window(memory_mask) 옵션, 영상 특징 차원을 설정에서 읽도록 변경, 오디오와 영상 프레임 수 확인"),
    ],
    "baseline/seld_challenge/loss.py": [
        ("2026-10-03", "원본에서 복사. 코드는 고치지 않음(이 이력 주석만 추가)"),
    ],
    "baseline/seld_challenge/main.py": [
        ("2026-10-03", "원본의 학습 진입점을 run.py의 train 단계로 넘기는 얇은 껍데기로 교체"),
    ],
    "baseline/seld_challenge/evaluate.py": [
        ("2026-10-03", "원본의 평가 데이터 예측을 run.py의 predict 단계로 넘기는 얇은 껍데기로 교체"),
    ],
    "baseline/seld_challenge/inference.py": [
        ("2026-10-03", "원본의 추론을 run.py의 eval 단계로 넘기는 얇은 껍데기로 교체"),
    ],
    "baseline/seld_challenge/config.py": [
        ("2026-10-03", "최초 생성: configs/base.yaml, 프리셋, --set 덮어쓰기를 합쳐 설정을 만들고 실행 이름, 라벨 길이, 배치 크기(auto)를 정함"),
        ("2026-10-07", "추가 학습 데이터 설정(train_stages, train_family, train_mirror, train_fraction, updates, audio_dtype, generated_dir), 데이터 꼬리표와 std_name 추가"),
    ],
    "baseline/seld_challenge/run.py": [
        ("2026-10-03", "최초 생성: 모든 단계(prepare, features, train, eval, swaptest, predict, score, summary)를 한 파일로"),
        ("2026-10-03", "소규모 시험 결과 반영: 제외 목록 평가를 _excl 파일로 따로 저장하고 summary를 제외 여부별로 계산, 학습 클립이 배치보다 적을 때 에러, summary의 격차 계산"),
        ("2026-10-07", "추가 학습 데이터(S1) 연동: 단계별 zip 풀기, 필터 선택, updates로 학습 길이 결정, datainfo 단계, 점수 파일에 데이터 구성 기록, summary에 data_gaps.csv"),
    ],
    "baseline/seld_challenge/data_generator.py": [
        ("2026-10-03", "원본에서 복사 후 재작성: fold 이름으로 파일을 찾던 것을 묶음 특징 파일 읽기와 split 이름으로 변경"),
        ("2026-10-07", "묶음을 합치지 않고 목록으로 읽는 PackedTensors, 단계(stages)와 keep 필터, audio_dtype=float16 추가"),
    ],
    "baseline/seld_challenge/extract_features.py": [
        ("2026-10-03", "원본에서 복사 후 재작성: 우리 데이터 구조에서 오디오, 영상, 라벨 특징을 500개씩 묶음 파일로 저장, video_crop=full(7x14) 추가, 프레임 수 검사"),
        ("2026-10-07", "단계별 클립 이름 필터와 묶음 접두사(s1train_000.pt), 영상 디코딩 스레드 미리 읽기"),
    ],
    "baseline/seld_challenge/check_scoring.py": [
        ("2026-10-03", "최초 생성: 채점 시험 3개(정답 그대로, 앞뒤 뒤집기, 빈 클립에 가짜 예측)"),
        ("2026-10-03", "(c) 시험의 출력을 소수 여섯째 자리까지 정확히 표시하도록 수정"),
    ],
    "baseline/seld_challenge/configs/base.yaml": [
        ("2026-10-03", "최초 생성: 모든 설정과 기본값(위치, 데이터셋, 모델 옵션, 학습, 평가)을 한국어 주석과 함께"),
        ("2026-10-07", "추가 학습 데이터 항목(updates, audio_dtype, generated_dir, train_stages, train_family, train_mirror, train_fraction) 추가"),
    ],
    "baseline/seld_challenge/configs/exclude_val_fix20.txt": [
        ("2026-10-03", "최초 생성: fix20에서 피크로 빠진 val_00088을 비교에서 뺄 때 쓰는 목록. 이후 변경 없음"),
    ],
    # ------------------------------------------------------------------ 노트북
    "baseline/seld_challenge/colab_run.ipynb": [
        ("2026-10-03", "최초 생성: 학습 노트북(prepare, features, train, eval, swaptest, predict, summary)"),
        ("2026-10-07", "추가 학습 데이터(S1)용으로 다시 만듦: 설정 칸 6개(train_stages, updates 등), datainfo 셀, data_gaps 출력, 추천 실험 순서. 이전 노트북은 build/colab_run_before_S1.ipynb"),
        ("2026-10-09", "사용자가 구분용으로 colab_run_new.ipynb로 바꿨다가, 이전 노트북이 build/colab_run_before_S1.ipynb로 옮겨져 구분이 필요 없어져 colab_run.ipynb로 되돌림(내용은 그대로)"),
    ],
    "build/colab_run_before_S1.ipynb": [
        ("2026-10-03", "colab_run.ipynb로 최초 생성(S0 전용 학습 노트북)"),
        ("2026-10-07", "S1 연동으로 colab_run.ipynb를 다시 만들기 전에 이전 내용을 이 이름으로 백업(내용은 그대로)"),
    ],
    "colab_generate.ipynb": [
        ("2026-10-07", "최초 생성: 추가 학습 데이터 생성 노트북(계획, 자체 점검, 묶음 생성/업로드/검증, 이어 하기)"),
        ("2026-10-07", "GPU 렌더러일 때 워커 수 상한(6) 추가"),
        ("2026-10-07", "코랩에서 실행되며 저장됨(실행 결과가 파일에 포함됨)"),
    ],
}

COMMENT_KINDS = {".py": "#", ".ps1": "#", ".sh": "#", ".yaml": "#", ".yml": "#", ".txt": "#"}


def entries_for(rel):
    h = HIST[rel]
    # 일괄 주석 줄은 이력 주석을 처음 붙인 날(TODAY)까지 만든 파일에만 붙인다. 그 뒤에 만들거나 고친 파일은 자기 이력 줄이 있다.
    return h if any(d > TODAY for d, _ in h) else h + [LAST]


def lines_for(rel):
    return [f"{d}  {t}" for d, t in entries_for(rel)]


def history_md(rel):
    """HTML 주석 블록(문자열). 노트북 생성 스크립트도 이것을 쓴다."""
    body = "\n".join("  " + l.replace("--", "- -") for l in lines_for(rel))
    return f"<!-- {START}\n{body}\n{END} -->"


def _eol(b):
    return "\r\n" if b"\r\n" in b else "\n"


def apply_hash(path, rel):
    b = open(path, "rb").read(); eol = _eol(b)
    bom = b"\xef\xbb\xbf" if b.startswith(b"\xef\xbb\xbf") else b""                              # PowerShell 스크립트 등은 맨 앞의 BOM을 그대로 둔다
    t = b[len(bom):].decode("utf-8")
    t = re.sub(r"(?m)^# \[변경 이력 시작\].*?^# \[변경 이력 끝\]\r?\n", "", t, flags=re.S)       # 기존 블록 제거(여러 줄)
    block = [f"# {START}"] + [f"#   {l}" for l in lines_for(rel)] + [f"# {END}"]
    lines = t.split("\n")
    at = 1 if lines and lines[0].startswith("#!") else 0                                        # 셔뱅 다음에
    new = lines[:at] + [x + ("\r" if eol == "\r\n" else "") for x in block] + lines[at:]
    open(path, "wb").write(bom + "\n".join(new).encode("utf-8"))


def apply_md(path, rel):
    b = open(path, "rb").read(); eol = _eol(b)
    bom = b"\xef\xbb\xbf" if b.startswith(b"\xef\xbb\xbf") else b""
    t = b[len(bom):].decode("utf-8")
    t = re.sub(r"\A<!-- \[변경 이력 시작\].*?\[변경 이력 끝\] -->\r?\n(\r?\n)?", "", t, flags=re.S)
    block = history_md(rel).replace("\n", eol)
    open(path, "wb").write(bom + (block + eol + eol + t).encode("utf-8"))


def apply_ipynb(path, rel):
    nb = json.load(open(path, encoding="utf8"))
    c0 = nb["cells"][0]
    assert c0["cell_type"] == "markdown", "첫 셀이 마크다운이 아닙니다"
    src = "".join(c0["source"])
    src = re.sub(r"\A<!-- \[변경 이력 시작\].*?\[변경 이력 끝\] -->\n\n?", "", src, flags=re.S)
    c0["source"] = (history_md(rel) + "\n\n" + src).splitlines(True)
    json.dump(nb, open(path, "w", encoding="utf8"), ensure_ascii=False, indent=1)


def main():
    done, missing = 0, []
    for rel in HIST:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            missing.append(rel); continue
        ext = os.path.splitext(rel)[1].lower()
        if ext == ".md": apply_md(path, rel)
        elif ext == ".ipynb": apply_ipynb(path, rel)
        elif ext in COMMENT_KINDS: apply_hash(path, rel)
        else: print("건너뜀(형식 모름):", rel); continue
        done += 1
    print(f"변경 이력 {done}개 파일에 적용, 없는 파일 {missing}")


if __name__ == "__main__":
    main()
