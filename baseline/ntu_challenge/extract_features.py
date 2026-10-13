# [변경 이력 시작]
#   2026-10-13  최초 생성: NTU 원본을 우리 폴더 구조에 맞게 다시 씀(묶음 파일 캐시, train 에서만 정규화 통계, train_NNNNN 만 사용)
#   2026-10-13  test 용으로 extract(labels=False) 추가(정답 없는 split)
# [변경 이력 끝]
"""
extract_features.py

NTU 저장소(itsjunwei/NTU_SNTL_Task3)의 extract_features.py 를 우리 데이터셋 구조에 맞게 다시 쓴 것이다.
특징을 계산하는 함수(utils.extract_stereo_features, utils.process_labels_adpit)는 원본 그대로 쓴다.

원본에서 바꾼 점
 - 입력 위치: {data_root}/{dataset}/{split}/{audio,labels}/  (원본은 stereo_dev/dev-*/, metadata_dev/dev-*/ 와 fold 이름)
 - 클립마다 .pt 파일을 만들지 않고 500개씩 묶은 파일(shard)로 저장한다(드라이브에서 작은 파일 수천 개를 읽으면 느리기 때문).
     {feat_dir}/audio/{split}_{000}.pt, {feat_dir}/labels_adpit/{split}_{000}.pt   각 파일은 {'names': [...], 'data': 텐서}
   이미 있는 묶음은 건너뛰므로 끊겨도 이어 할 수 있다.
 - 정규화 통계(채널별, 주파수별 평균과 표준편차)는 **train 에서만** 구한다. 원본은 test fold 까지 포함한 전체 dev 특징으로 구했다.
   StandardScaler 와 같은 계산이다(왼쪽, 오른쪽 채널은 통계를 공유해서 채널 사이 차이를 보존). 정규화는 불러올 때 적용한다.
 - 오디오 시간 프레임 수가 라벨 프레임 수와 다르면 에러를 낸다.
"""

import os
import re
import glob
import math
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import utils

SHARD = 500


S0_TRAIN = re.compile(r'^train_\d+$')


def clip_names(dir_, ext, split):
    """폴더의 클립 이름(확장자 제외). train 폴더에는 추가 생성 데이터(train_s1_NNNNN 등)가 같이 풀려 있을 수 있어서, 이 베이스라인은 기본 데이터(train_NNNNN)만 쓴다."""
    names = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(dir_, '*' + ext)))
    return [n for n in names if S0_TRAIN.match(n)] if split == 'train' else names


def shard_path(feat_dir, kind, split, i):
    return os.path.join(feat_dir, kind, f'{split}_{i:03d}.pt')


def save_shard(path, names, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'names': names, 'data': data}, path + '.tmp')
    os.replace(path + '.tmp', path)


def n_shared_channels(params):
    """왼쪽, 오른쪽(그리고 mid, side) 채널은 통계를 공유한다."""
    return 4 if params['ms'] else 2


class SELDFeatureExtractor():
    def __init__(self, params):
        self.params = params
        self.data_root = os.path.join(params['data_root'], params['dataset'])
        self.feat_dir = params['feat_dir']

        # audio feature extraction (원본과 같은 계산)
        self.sampling_rate = params['sampling_rate']
        self.hop_length = int(self.sampling_rate * params['hop_length_s'])
        self.n_fft = 2 ** (2 * self.hop_length - 1).bit_length()
        self.win_length = self.n_fft
        self.nb_mels = params['nb_mels']
        if self.nb_mels == 0:   # Cutdown to 200 Freq bins for linspec
            self.n_fft = 512
            self.win_length = 512
        self.max_freq = params['max_freq']
        self.t_pool = int(np.prod(params['t_pool_size']))
        self.nb_label_frames = params['label_sequence_length']
        self.nb_unique_classes = params['nb_classes']

    # ------------------------------------------------------------------ 오디오
    def _audio_feature(self, path):
        audio, sr = utils.load_audio(path, self.sampling_rate)
        f = utils.extract_stereo_features(audio, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                                          nb_mels=self.nb_mels, max_freq=self.max_freq, use_gamma=self.params['gamma'],
                                          use_ipd=self.params['ipd'], use_iv=self.params['iv'], use_slite=self.params['slite'],
                                          use_ms=self.params['ms'])
        if f.shape[1] // self.t_pool != self.nb_label_frames:
            raise ValueError(f"오디오 시간 프레임 {f.shape[1]} (풀링 후 {f.shape[1] // self.t_pool}) != 라벨 프레임 {self.nb_label_frames}: {path}")
        return torch.tensor(f, dtype=torch.float32)

    def extract_audio_features(self, split, threads=4):
        adir = os.path.join(self.data_root, split, 'audio')
        names = clip_names(adir, '.wav', split)
        if not names:
            raise FileNotFoundError(f"오디오가 없습니다: {adir}")
        for i in range(math.ceil(len(names) / SHARD)):
            path = shard_path(self.feat_dir, 'audio', split, i)
            if os.path.exists(path):
                continue
            chunk = names[i * SHARD:(i + 1) * SHARD]
            with ThreadPoolExecutor(threads) as ex:
                feats = list(tqdm(ex.map(lambda n: self._audio_feature(os.path.join(adir, n + '.wav')), chunk),
                                  total=len(chunk), desc=f'audio {split} shard {i}', unit='clip'))
            save_shard(path, chunk, torch.stack(feats))

    # ------------------------------------------------------------------ 라벨
    def extract_labels(self, split):
        ldir = os.path.join(self.data_root, split, 'labels')
        names = clip_names(os.path.join(self.data_root, split, 'audio'), '.wav', split)
        for i in range(math.ceil(len(names) / SHARD)):
            path = shard_path(self.feat_dir, 'labels_adpit', split, i)
            if os.path.exists(path):
                continue
            chunk = names[i * SHARD:(i + 1) * SHARD]
            out = []
            for n in tqdm(chunk, desc=f'labels {split} shard {i}', unit='clip'):
                label_data = utils.load_labels(os.path.join(ldir, n + '.csv'))
                out.append(utils.process_labels_adpit(label_data, self.nb_label_frames, self.nb_unique_classes, params=self.params))
            save_shard(path, chunk, torch.stack(out))

    # ------------------------------------------------------------------ 정규화 통계 (train 에서만)
    def stats_path(self):
        return os.path.join(self.feat_dir, 'stats_train.npz')

    def fit_stats(self):
        """train 오디오 특징으로 (채널, 주파수)별 평균과 표준편차를 구해 저장한다. StandardScaler 와 같은 값(분산 0 이면 표준편차 1)."""
        path = self.stats_path()
        if os.path.exists(path):
            return
        files = sorted(glob.glob(os.path.join(self.feat_dir, 'audio', 'train_*.pt')))
        if not files:
            raise FileNotFoundError('train 오디오 특징이 없습니다. 먼저 extract_audio_features(train)')
        k = n_shared_channels(self.params)
        s = s2 = n = None
        for f in tqdm(files, desc='정규화 통계(train)', unit='shard'):
            x = torch.load(f)['data'].double()                   # (N, C, T, F)
            if s is None:
                C, F = x.shape[1], x.shape[3]
                s, s2, n = np.zeros((C, F)), np.zeros((C, F)), np.zeros(C)
            s += x.sum(dim=(0, 2)).numpy()
            s2 += (x ** 2).sum(dim=(0, 2)).numpy()
            n += x.shape[0] * x.shape[2]
            del x
        # 앞의 k 채널은 합쳐서 같은 통계를 쓴다
        s[:k] = s[:k].sum(0); s2[:k] = s2[:k].sum(0); n[:k] = n[:k].sum()
        mean = s / n[:, None]
        std = np.sqrt(np.maximum(s2 / n[:, None] - mean ** 2, 0))
        std[std == 0] = 1.0
        np.savez(path + '.tmp.npz', mean=mean.astype(np.float32), std=std.astype(np.float32))
        os.replace(path + '.tmp.npz', path)

    # ------------------------------------------------------------------ 한꺼번에
    def extract(self, splits=('train', 'val'), labels=True):
        """labels=False 는 test(정답 없음)용: 오디오 특징만 뽑는다. 정규화 통계는 train 것이 있어야 한다."""
        for sp in splits:
            self.extract_audio_features(sp)
            if labels:
                self.extract_labels(sp)
        self.fit_stats()
