# [변경 이력 시작]
#   2026-10-13  최초 생성: NTU 원본 parameters.py에서 가져와 정리(경로, 데이터셋, 라벨 길이 auto, accum_batch auto, 선택 기준, 특징과 증강 옵션 기본값). 사용하지 않는 fold, wandb, finetune 항목 제거
# [변경 이력 끝]
"""
parameters.py

모든 기본값. NTU 저장소(itsjunwei/NTU_SNTL_Task3)의 parameters.py 에서 가져와 우리 데이터셋에 맞게 정리했다.
명령줄 인자(main.py)가 이 값을 덮어쓴다. 바꾼 점은 README_ntu.md 의 '원본에서 바꾼 곳' 참고.
"""

params = {

    # choose task (NTU 코드는 소리만 쓴다)
    'modality': 'audio',
    'net_type': 'SELDnet',
    'model': 'seldnet',

    # 데이터 위치 (main.py 가 채운다)
    'data_root': '',            # {data_root}/{dataset}/{train,val}/{audio,labels}/ 가 있는 곳 (zip 을 풀어 놓는 곳)
    'out_root': '',             # 결과를 저장할 곳: {out_root}/runs/{run_name}, {out_root}/features/ntu_{...}
    'dataset': 'main20',        # main20 | ctrl5
    'feat_dir': '',

    # audio feature extraction params (NTU 원본과 같음)
    'sampling_rate': 24000,
    'hop_length_s': 0.0125,     # 12.5ms -> 라벨 프레임(100ms) 하나가 STFT 프레임 8개
    'nb_mels': 64,
    'max_freq': 2000,

    # model params (NTU 원본과 같음)
    'nb_conv_blocks': 3,
    'nb_conv_filters': 64,
    'f_pool_size': [4, 4, 2],
    't_pool_size': [2, 2, 2],
    'dropout': 0.05,

    'rnn_size': 128,
    'nb_rnn_layers': 2,

    'nb_self_attn_layers': 2,
    'nb_attn_heads': 8,

    'nb_fnn_layers': 1,
    'fnn_size': 128,

    'max_polyphony': 3,   # tracks for multiaccdoa
    'nb_classes': 13,
    'label_sequence_length': 200,   # 데이터셋에 따라 main.py 가 정한다: main20 200, ctrl5 50 (100ms 프레임)

    # loss params
    'multiACCDOA': True,
    'thresh_unify': 30,

    # training params (NTU 원본: 100에폭, AdamW, OneCycleLR)
    'nb_epochs': 100,
    'batch_size': 32,           # 한 번에 GPU 에 올리는 클립 수
    'accum_batch': 'auto',      # 가중치를 갱신하는 단위(클립 수). auto = 12800 / 라벨 프레임 수 (main20 64, ctrl5 256)
    'val_batch_size': 32,
    'nb_workers': 2,
    'shuffle': True,
    'seed': 1,

    # optimizer params
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'val_freq': 2,              # val 점수를 몇 에폭마다 내나 (마지막 20% 구간은 매 에폭)

    # 모델 고르는 기준 (val 점수): F_ext = 방위각을 접지 않음(챌린지 점수), F_basic = 접음(DCASE 공식과 같음)
    'select_metric': 'F_ext',
    'exclude_absent_classes': True,

    # metric params (채점: score.py)
    'average': 'macro',
    'lad_doa_thresh': 20,
    'lad_dist_thresh': float('inf'),
    'lad_reldist_thresh': 1.0,
    'lad_req_onscreen': False,

    # Distance Normalization Variables (NTU 원본 값. 우리 train 라벨 거리의 평균 1.95m, 표준편차 0.68m 와 비슷하다)
    'dnorm': True,
    'd_mean': 1.9242867099100318,
    'd_std': 0.663059233462568,
    'd_max': 7.685155462626762,

    # 특징과 증강 옵션 (NTU 원본 기능. 기본은 모두 꺼짐 = 로그멜 L/R 두 채널)
    'gamma': False, 'ipd': False, 'iv': False, 'slite': False, 'ms': False,
    'cutout': False, 'itfm': False, 'freqshift': False, 'filtaug': False, 'compfreq': False, 'alltrans': False,
}
