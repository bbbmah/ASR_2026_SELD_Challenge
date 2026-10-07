# [변경 이력 시작]
#   2026-10-03  원본에서 복사 후 재작성: 우리 데이터 구조에서 오디오, 영상, 라벨 특징을 500개씩 묶음 파일로 저장, video_crop=full(7x14) 추가, 프레임 수 검사
#   2026-10-07  단계별 클립 이름 필터와 묶음 접두사(s1train_000.pt), 영상 디코딩 스레드 미리 읽기
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
extract_features.py

오디오(로그 멜), 영상(ResNet-50), 라벨(ADPIT) 특징을 뽑아 `feat_dir` 에 캐시한다. 원본(Parthasaarathy Sudarsanam)에서 바꾼 점:
 - 입력 위치가 우리 데이터셋 구조 `{data_root}/{dataset}/{split}/{audio,video,labels}/` 이다.
 - 클립마다 .pt 파일을 만들지 않고 500개씩 묶은 파일(shard)로 저장한다(드라이브에서 작은 파일 수천 개를 읽으면 느리기 때문).
   저장 위치: {feat_dir}/audio/{key}_{000}.pt, {feat_dir}/video_{center|full}/{key}_{000}.pt, {feat_dir}/labels_adpit/{key}_{000}.pt
   각 파일은 {'names': [클립 이름...], 'data': 텐서} 이다. 이미 있는 묶음 파일은 건너뛰므로 끊겨도 이어 할 수 있다.
 - 추가 학습 데이터(단계 1 이상): 클립 이름이 train_s{단계}_NNNNN 이고 같은 train 폴더에 풀린다. 단계 0 은 train_NNNNN 만,
   단계 k 는 train_s{k}_ 로 시작하는 것만 골라서 뽑고, 묶음 파일 접두사(key)는 단계 0 'train', 단계 k 's{k}train' 이다.
 - 오디오 특징은 영상 옵션과 무관하게 한 번만 뽑고, 영상 특징은 video_crop 별로 따로 저장한다.
 - video_crop='full': 화면을 자르지 않고 224x448 로 줄여 ResNet 에 넣는다(특징 지도 7x14, 채널 평균 후 98차원).
 - 영상 프레임 수가 라벨 프레임 수와, 오디오 시간 프레임 수가 라벨 프레임 수와 다르면 에러를 낸다.
 - 영상 디코딩을 여러 스레드로 미리 읽어 GPU 계산과 겹친다(클립이 수천 개일 때 오래 걸리지 않게).
"""

import os
import re
import glob
import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from tqdm import tqdm
import utils
from data_generator import stage_key

SHARD = 500
S0_TRAIN = re.compile(r'^train_\d+$')


def clip_names(dir_, ext, split='val', stage=0):
    """폴더의 클립 이름(확장자 제외). train 은 단계에 맞는 이름만 고른다."""
    names = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(dir_, '*' + ext)))
    if split == 'train':
        names = [n for n in names if (S0_TRAIN.match(n) if int(stage) == 0 else n.startswith(f'train_s{int(stage)}_'))]
    return names


def shard_path(feat_dir, kind, key, i):
    return os.path.join(feat_dir, kind, f'{key}_{i:03d}.pt')


def save_shard(path, names, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'names': names, 'data': data}, path + '.tmp')
    os.replace(path + '.tmp', path)


class SELDFeatureExtractor():
    def __init__(self, params, data_dirs, feat_dir):
        """
        params: config.py 가 만든 params.  data_dirs: config.dataset_dirs() 결과.  feat_dir: 이 데이터셋의 특징 캐시 폴더.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.params = params
        self.dirs = data_dirs
        self.feat_dir = feat_dir

        # audio feature extraction
        self.sampling_rate = params['sampling_rate']
        self.hop_length = int(self.sampling_rate * params['hop_length_s'])
        self.win_length = 2 * self.hop_length
        self.n_fft = 2 ** (self.win_length - 1).bit_length()
        self.nb_mels = params['nb_mels']
        self.t_pool = int(np.prod(params['t_pool_size']))

        self.fps = params['fps']
        self.nb_label_frames = params['label_sequence_length']
        self.nb_unique_classes = params['nb_classes']
        self._resnet = None

    # ------------------------------------------------------------------ 오디오
    def extract_audio_features(self, split, stage=0):
        names = clip_names(self.dirs['audio'](split), '.wav', split, stage); key = stage_key(split, stage)
        for i in range(math.ceil(len(names) / SHARD)):
            path = shard_path(self.feat_dir, 'audio', key, i)
            if os.path.exists(path):
                continue
            chunk = names[i * SHARD:(i + 1) * SHARD]
            feats = []
            for n in tqdm(chunk, desc=f"audio {key} shard {i}", unit="clip"):
                audio, sr = utils.load_audio(os.path.join(self.dirs['audio'](split), n + '.wav'), self.sampling_rate)
                f = utils.extract_log_mel_spectrogram(audio, sr, self.n_fft, self.hop_length, self.win_length, self.nb_mels)
                if f.shape[1] // self.t_pool != self.nb_label_frames:
                    raise ValueError(f"오디오 시간 프레임 {f.shape[1]} (풀링 후 {f.shape[1] // self.t_pool}) != 라벨 프레임 {self.nb_label_frames}: {n}")
                feats.append(torch.tensor(f, dtype=torch.float32))
            save_shard(path, chunk, torch.stack(feats))

    # ------------------------------------------------------------------ 영상
    def _load_resnet(self, crop):
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.DEFAULT
        if self._resnet is None:
            model = resnet50(weights=weights).to(self.device)
            self._resnet = torch.nn.Sequential(*(list(model.children())[:-2])).eval()
        if crop == 'center':
            return weights.transforms(), (360, 180)             # 원래 전처리: 360x180 -> 짧은 변 232 -> 가운데 224x224 자르기
        mean = torch.tensor(weights.transforms().mean).view(3, 1, 1)
        std = torch.tensor(weights.transforms().std).view(3, 1, 1)

        def full_preprocess(img):                                # 자르지 않는다: 이미 448x224 로 줄여 둔 PIL 이미지
            x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.
            return (x - mean) / std
        return full_preprocess, (448, 224)

    def extract_video_features(self, split, crop, stage=0, threads=4):
        names = clip_names(self.dirs['video'](split), '.mp4', split, stage)
        key = stage_key(split, stage); kind = f'video_{crop}'
        pre, size = None, None
        for i in range(math.ceil(len(names) / SHARD)):
            path = shard_path(self.feat_dir, kind, key, i)
            if os.path.exists(path):
                continue
            if pre is None:
                pre, size = self._load_resnet(crop)
            chunk = names[i * SHARD:(i + 1) * SHARD]
            load = lambda n: utils.load_video(os.path.join(self.dirs['video'](split), n + '.mp4'), self.fps,
                                              expected_frames=self.nb_label_frames, size=size)
            feats = []
            with ThreadPoolExecutor(threads) as ex:              # 디코딩(스레드)과 ResNet(GPU)을 겹친다. map 은 순서를 지킨다.
                for frames in tqdm(ex.map(load, chunk), total=len(chunk), desc=f"video({crop}) {key} shard {i}", unit="clip"):
                    feats.append(utils.extract_resnet_features(frames, pre, self._resnet, self.device))
            save_shard(path, chunk, torch.stack(feats))

    # ------------------------------------------------------------------ 라벨
    def extract_labels(self, split, stage=0):
        names = clip_names(self.dirs['labels'](split), '.csv', split, stage); key = stage_key(split, stage)
        kind = 'labels_adpit' if self.params['multiACCDOA'] else 'labels_single'
        for i in range(math.ceil(len(names) / SHARD)):
            path = shard_path(self.feat_dir, kind, key, i)
            if os.path.exists(path):
                continue
            chunk = names[i * SHARD:(i + 1) * SHARD]
            out = []
            for n in chunk:
                label_data = utils.load_labels(os.path.join(self.dirs['labels'](split), n + '.csv'))
                if self.params['multiACCDOA']:
                    out.append(utils.process_labels_adpit(label_data, self.nb_label_frames, self.nb_unique_classes))
                else:
                    out.append(utils.process_labels(label_data, self.nb_label_frames, self.nb_unique_classes))
            save_shard(path, chunk, torch.stack(out))

    # ------------------------------------------------------------------ 한꺼번에
    def extract_features(self, split, stage=0):
        """modality 에 맞춰 오디오(항상), 영상(audio_visual 일 때)을 뽑는다."""
        self.extract_audio_features(split, stage)
        if self.params['modality'] == 'audio_visual':
            self.extract_video_features(split, self.params['video_crop'], stage)
        if os.path.isdir(self.dirs['labels'](split)):
            self.extract_labels(split, stage)
