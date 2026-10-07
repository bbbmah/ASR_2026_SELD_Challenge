# [변경 이력 시작]
#   2026-10-03  원본에서 복사 후 수정: 데이터 경로, fold 이름, 라벨 길이, 배치 크기 등을 configs/base.yaml과 config.py로 옮기고 새 옵션 키(video_input, video_crop, av_time_window 등) 추가
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
parameters.py

모델과 학습의 기본값. 원본(Parthasaarathy Sudarsanam, Tampere University)에서 데이터 경로, fold 이름, label_sequence_length,
batch_size 등을 설정 파일(configs/base.yaml)과 config.py로 옮겼다. 여기 있는 값은 config.py가 읽어서
설정 값으로 덮어쓴 뒤 `params` 딕셔너리로 모델, 데이터 로더, 채점에 넘긴다.
"""

params = {

    # choose task (아래 값은 config.py가 프리셋/설정으로 덮어쓴다)
    'modality': 'audio_visual',  # 'audio' or 'audio_visual'
    'net_type': 'SELDnet',
    'video_input': 'real',       # 'real' | 'zeros'
    'video_crop': 'center',      # 'center' | 'full'
    'av_time_window': None,      # None | 정수 k

    # audio feature extraction params
    'sampling_rate': 24000,
    'hop_length_s': 0.02,
    'nb_mels': 64,

    # video feature extraction params
    'fps': 10,
    'resnet_feature_size': 49,  # center: (7,7) -> 49,  full: (7,14) -> 98 (config.py가 정한다)

    # model params
    'nb_conv_blocks': 3,
    'nb_conv_filters': 64,
    'f_pool_size': [4, 4, 2],
    't_pool_size': [5, 1, 1],
    'dropout': 0.05,

    'rnn_size': 128,
    'nb_rnn_layers': 2,

    'nb_self_attn_layers': 2,
    'nb_attn_heads': 8,

    'nb_transformer_layers': 2,

    'nb_fnn_layers': 1,
    'fnn_size': 128,

    'max_polyphony': 3,   # tracks for multiaccdoa
    'nb_classes': 13,
    'label_sequence_length': 200,  # config.py가 데이터셋에 따라 정한다: main20/fix20 200, ctrl5 50

    # loss params
    'multiACCDOA': True,
    'thresh_unify': 15,

    # training params (config.py가 설정 값으로 덮어쓴다)
    'nb_epochs': 200,
    'batch_size': 64,
    'nb_workers': 2,
    'shuffle': True,

    # optimizer params
    'learning_rate': 1e-3,
    'weight_decay': 0,

    # metric params
    'average': 'macro',                  # Supports 'micro': sample-wise average and 'macro': class-wise average.
    'lad_doa_thresh': 20,                # DOA error threshold for computing the detection metrics.
    'lad_dist_thresh': float('inf'),     # Absolute distance error threshold for computing the detection metrics.
    'lad_reldist_thresh': float('1.0'),  # Relative distance error threshold for computing the detection metrics.
    'lad_req_onscreen': False,           # Require correct on-screen estimation when computing the detection metrics.
    'exclude_absent_classes': True,      # 정답이 하나도 없는 클래스를 평균 F에서 뺀다
}
