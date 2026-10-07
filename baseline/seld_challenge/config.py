# [변경 이력 시작]
#   2026-10-03  최초 생성: configs/base.yaml, 프리셋, --set 덮어쓰기를 합쳐 설정을 만들고 실행 이름, 라벨 길이, 배치 크기(auto)를 정함
#   2026-10-07  추가 학습 데이터 설정(train_stages, train_family, train_mirror, train_fraction, updates, audio_dtype, generated_dir), 데이터 꼬리표와 std_name 추가
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
config.py

configs/base.yaml + 프리셋 + 명령줄 덮어쓰기(--set key=value)를 합쳐서 하나의 설정(Config)을 만든다.
Config.params 는 원본 코드가 쓰던 params 딕셔너리(모델, 데이터 로더, 채점용)와 같은 모양이다.
"""

import os
import yaml
from parameters import params as DEFAULT_PARAMS

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_YAML = os.path.join(HERE, 'configs', 'base.yaml')

# 프리셋: 개별 옵션이 비어 있을 때 쓰는 값
PRESETS = {
    'B1':   dict(modality='audio',        video_input='real',  video_crop='center', av_time_window='none'),
    'B2':   dict(modality='audio_visual', video_input='real',  video_crop='center', av_time_window='none'),
    'B2-0': dict(modality='audio_visual', video_input='zeros', video_crop='center', av_time_window='none'),
    'B3':   dict(modality='audio_visual', video_input='real',  video_crop='full',   av_time_window=2),
}
OPTION_KEYS = ('modality', 'video_input', 'video_crop', 'av_time_window')

LABEL_LEN = {'main20': 200, 'fix20': 200, 'ctrl5': 50}     # 라벨 프레임 수(100ms)
SPLITS = {'main20': ('train', 'val', 'test'), 'ctrl5': ('train', 'val', 'test'), 'fix20': ('val', 'test')}
FRAMES_PER_BATCH = 12800                                     # batch_size=auto 일 때: 12800 / 라벨 프레임 수

# 실행 이름 뒤에 붙일 짧은 이름 (프리셋/기본값과 다르게 덮어썼을 때만)
SUFFIX_NAMES = dict(modality='mod', video_input='vi', video_crop='vc', av_time_window='win', epochs='ep', lr='lr',
                    batch_size='bs', select_metric='sel', exclude_absent_classes='exabs')

STAGES = ('prepare', 'features', 'train', 'eval', 'swaptest', 'predict', 'score', 'summary')


def _blank(v):
    return v is None or (isinstance(v, str) and v.strip() == '')


def parse_sets(sets):
    """['seed=2', 'lr=0.0005'] -> {'seed': 2, 'lr': 0.0005}  (값은 YAML 규칙으로 해석)"""
    out = {}
    for item in sets or []:
        if '=' not in item:
            raise ValueError(f"--set 은 key=value 형태여야 합니다: {item}")
        k, v = item.split('=', 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def parse_stages(v):
    """0, "0,1", [0, 1] -> [0, 1]  (정렬, 중복 제거, 0 이상의 정수)"""
    if isinstance(v, (list, tuple)):
        items = list(v)
    else:
        items = [x for x in str(v).replace(' ', '').split(',') if x != '']
    out = sorted({int(x) for x in items})
    if not out or min(out) < 0:
        raise ValueError(f"train_stages 는 0 이상의 정수 목록이어야 합니다: {v}")
    return out


class Config(dict):
    """dict 이면서 cfg.key 로도 읽을 수 있다."""
    __getattr__ = dict.__getitem__


def load_config(dataset=None, preset=None, sets=None, base_yaml=BASE_YAML, need_drive_root=True):
    with open(base_yaml, encoding='utf8') as f:
        base = yaml.safe_load(f)
    cfg = dict(base)
    over = parse_sets(sets)
    if dataset is not None:
        over['dataset'] = dataset
    if preset is not None:
        over['preset'] = preset
    unknown = [k for k in over if k not in base]
    if unknown:
        raise KeyError(f"알 수 없는 설정 이름: {unknown}  (가능한 이름은 configs/base.yaml 참조)")
    cfg.update(over)

    if cfg['preset'] not in PRESETS:
        raise ValueError(f"preset 은 {list(PRESETS)} 중 하나여야 합니다: {cfg['preset']}")
    if cfg['dataset'] not in ('main20', 'ctrl5'):
        raise ValueError("dataset(학습 데이터셋)은 main20 또는 ctrl5 여야 합니다. fix20 은 eval_dataset 으로 쓰세요.")
    if _blank(cfg['eval_dataset']):
        cfg['eval_dataset'] = cfg['dataset']
    if cfg['eval_dataset'] not in LABEL_LEN:
        raise ValueError(f"eval_dataset 은 {list(LABEL_LEN)} 중 하나여야 합니다.")

    # 개별 옵션이 비어 있으면 프리셋 값, 있으면 그 값. 실행 이름에 붙일 접미사도 여기서 만든다.
    preset_vals = PRESETS[cfg['preset']]
    suffix = []
    for k in OPTION_KEYS:
        if _blank(cfg[k]):
            cfg[k] = preset_vals[k]
        elif str(cfg[k]) != str(preset_vals[k]):
            suffix.append(f"{SUFFIX_NAMES[k]}{cfg[k]}")
    cfg['av_time_window'] = None if str(cfg['av_time_window']).lower() == 'none' else int(cfg['av_time_window'])
    if cfg['modality'] not in ('audio', 'audio_visual'):
        raise ValueError("modality 는 audio 또는 audio_visual")
    if cfg['video_input'] not in ('real', 'zeros') or cfg['video_crop'] not in ('center', 'full'):
        raise ValueError("video_input 은 real|zeros, video_crop 은 center|full")
    if cfg['eval_video'] not in ('real', 'zeros', 'shuffle'):
        raise ValueError("eval_video 는 real|zeros|shuffle")
    if cfg['select_metric'] not in ('F_ext', 'F_basic'):
        raise ValueError("select_metric 은 F_ext|F_basic")
    # 추가 학습 데이터 설정과 "데이터 꼬리표"(실행 이름에 들어간다. 기존 S0 실행의 이름은 그대로)
    cfg['train_stages'] = parse_stages(cfg['train_stages'])
    cfg['updates'] = None if _blank(cfg['updates']) else int(cfg['updates'])
    cfg['audio_dtype'] = str(cfg['audio_dtype']).lower()
    if cfg['audio_dtype'] not in ('float32', 'float16'):
        raise ValueError("audio_dtype 는 float32|float16")
    extra = any(s > 0 for s in cfg['train_stages'])
    fam = [x.strip() for x in str(cfg['train_family']).split(',') if x.strip()] if not _blank(cfg['train_family']) else []
    mir = None if _blank(cfg['train_mirror']) else int(cfg['train_mirror'])
    frac = float(cfg['train_fraction'])
    if not 0 < frac <= 1:
        raise ValueError("train_fraction 은 0 보다 크고 1 이하")
    if mir not in (None, 0, 1):
        raise ValueError("train_mirror 는 비움, 0, 1")
    bad_fam = [f for f in fam if f not in ('F1', 'F2', 'F3', 'F4', 'F5')]
    if bad_fam:
        raise ValueError(f"train_family 는 F1~F5 중에서: {bad_fam}")
    if not extra:                                          # 단계 0 만 쓰면 필터는 의미가 없으므로 무시한다
        fam, mir, frac = [], None, 1.0
    cfg['train_family'], cfg['train_mirror'], cfg['train_fraction'] = fam, mir, frac
    tag = []
    if cfg['train_stages'] != [0]:
        tag.append('S' + ''.join(str(s) for s in cfg['train_stages']))
    if fam: tag.append('fam' + ''.join(fam))
    if mir is not None: tag.append(f'mir{mir}')
    if frac != 1.0: tag.append(f'fr{frac:g}')
    if cfg['updates'] is not None: tag.append(f"up{cfg['updates']}")
    if cfg['audio_dtype'] == 'float16': tag.append('f16')
    cfg['data_tag'] = '-'.join(tag)

    for k in ('epochs', 'lr', 'batch_size', 'select_metric', 'exclude_absent_classes'):
        if k == 'epochs' and cfg['updates'] is not None:   # updates 를 쓰면 에폭 수는 거기서 정해지므로 꼬리표에 넣지 않는다
            continue
        if str(cfg[k]) != str(base[k]):
            suffix.append(f"{SUFFIX_NAMES[k]}{cfg[k]}")
    cfg['std_name'] = f"{cfg['dataset']}_{cfg['preset']}_s{cfg['seed']}" + (f"_{cfg['data_tag']}" if cfg['data_tag'] else "")
    cfg['run_name'] = cfg['std_name'] + ("_" + "-".join(suffix) if suffix else "")

    # 라벨 길이와 배치 크기
    cfg['label_len'] = LABEL_LEN[cfg['dataset']]
    cfg['eval_label_len'] = LABEL_LEN[cfg['eval_dataset']]
    if str(cfg['batch_size']).lower() == 'auto':
        cfg['batch_size_train'] = FRAMES_PER_BATCH // cfg['label_len']
    else:
        cfg['batch_size_train'] = int(cfg['batch_size'])
    cfg['batch_size_eval'] = cfg['batch_size_train'] if str(cfg['batch_size']).lower() != 'auto' \
        else FRAMES_PER_BATCH // cfg['eval_label_len']

    # 경로
    if need_drive_root and _blank(cfg['drive_root']):
        raise ValueError("drive_root 를 지정하세요 (예: --set drive_root=/content/drive/MyDrive/.../for_dataset)")
    dr = cfg['drive_root'] or ''
    cfg['out_root'] = cfg['out_root'] if not _blank(cfg['out_root']) else os.path.join(dr, 'seld_runs')
    cfg['public_zip_dir'] = cfg['public_zip_dir'] if not _blank(cfg['public_zip_dir']) else os.path.join(dr, 'dataset', 'main20')
    cfg['internal_zip_dir'] = cfg['internal_zip_dir'] if not _blank(cfg['internal_zip_dir']) else os.path.join(dr, 'internal')
    cfg['run_dir'] = os.path.join(cfg['out_root'], 'runs', cfg['run_name'])
    cfg['feat_root'] = os.path.join(cfg['out_root'], 'features')
    cfg['results_dir'] = os.path.join(cfg['out_root'], 'results')
    cfg['generated_dir'] = cfg['generated_dir'] if not _blank(cfg['generated_dir']) else os.path.join(dr, 'generated')

    # 원본 코드가 쓰는 params 딕셔너리
    p = dict(DEFAULT_PARAMS)
    p.update(modality=cfg['modality'], video_input=cfg['video_input'], video_crop=cfg['video_crop'],
             av_time_window=cfg['av_time_window'], label_sequence_length=cfg['label_len'],
             nb_epochs=int(cfg['epochs']), learning_rate=float(cfg['lr']), batch_size=cfg['batch_size_train'],
             nb_workers=int(cfg['num_workers']), exclude_absent_classes=bool(cfg['exclude_absent_classes']),
             resnet_feature_size=49 if cfg['video_crop'] == 'center' else 98, audio_dtype=cfg['audio_dtype'])
    cfg['params'] = p
    return Config(cfg)


def dataset_dirs(cfg, dataset):
    """로컬에 풀어 둔 데이터셋의 폴더들."""
    base = os.path.join(cfg['data_root'], dataset)
    return dict(base=base,
                audio=lambda sp: os.path.join(base, sp, 'audio'),
                video=lambda sp: os.path.join(base, sp, 'video'),
                labels=lambda sp: os.path.join(base, sp, 'labels'),
                test_labels=os.path.join(base, 'private', 'test_labels'))


def feat_dir(cfg, dataset):
    return os.path.join(cfg['feat_root'], dataset)
