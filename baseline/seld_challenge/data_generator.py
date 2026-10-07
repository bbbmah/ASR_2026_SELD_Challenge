# [변경 이력 시작]
#   2026-10-03  원본에서 복사 후 재작성: fold 이름으로 파일을 찾던 것을 묶음 특징 파일 읽기와 split 이름으로 변경
#   2026-10-07  묶음을 합치지 않고 목록으로 읽는 PackedTensors, 단계(stages)와 keep 필터, audio_dtype=float16 추가
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
data_generator.py

캐시된 특징(extract_features.py 가 만든 묶음 파일)을 메모리에 올려서 학습/평가용 데이터를 돌려준다.
원본(Parthasaarathy Sudarsanam)에서 fold 이름으로 파일을 찾던 부분을 split 이름('train', 'val', 'test')과 묶음 파일 읽기로 바꿨다.

추가 학습 데이터(S1 등)를 쓰기 위해 아래를 더했다.
 - 묶음 파일을 하나의 큰 텐서로 합치지 않고 **목록으로 들고** 인덱스로 읽는다(PackedTensors). 합치면 메모리가 두 배로 잠깐 필요해서
   S0+S1(약 9천 클립, 약 8GB)에서 램이 모자랄 수 있다.
 - 단계별 묶음 파일 접두사: 단계 0 은 'train', 단계 k>=1 은 's{k}train' (예: s1train_000.pt). 'train_*.pt' 패턴에 안 걸려서
   S0 만 학습할 때 새 단계가 섞이지 않는다.
 - keep(클립 이름 집합)으로 가족, 거울, 비율 필터를 적용한다(필터에서 빠진 클립은 메모리에 올리지 않는다).
 - audio_dtype='float16' 이면 오디오 특징을 반정밀도로 들고 있다가 꺼낼 때 float32 로 바꾼다(메모리 절반).
"""

import os
import glob
import bisect
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


def stage_key(split, stage):
    """묶음 파일 접두사. 단계 0 = 'train', 단계 1 = 's1train'. (val, test 는 단계 0 만 있다)"""
    return split if int(stage) == 0 else f's{int(stage)}{split}'


class PackedTensors:
    """묶음 텐서 여러 개를 합치지 않고 하나처럼 인덱스로 읽는다."""
    def __init__(self, parts=None, out_dtype=None):
        self.parts = list(parts or [])
        self.out_dtype = out_dtype
        self.offsets = [0]
        for p in self.parts:
            self.offsets.append(self.offsets[-1] + int(p.shape[0]))

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, i):
        i = int(i)
        k = bisect.bisect_right(self.offsets, i) - 1
        x = self.parts[k][i - self.offsets[k]]
        return x.to(self.out_dtype) if (self.out_dtype is not None and x.dtype != self.out_dtype) else x

    def to_tensor(self):
        """하나의 텐서로 합친다(작은 split 에서만 쓴다)."""
        x = torch.cat(self.parts, dim=0)
        return x.to(self.out_dtype) if self.out_dtype is not None else x

    @staticmethod
    def from_tensor(t, out_dtype=None):
        return PackedTensors([t], out_dtype)

    def zeros_like(self):
        return PackedTensors([torch.zeros_like(p) for p in self.parts], self.out_dtype)


def load_packed(feat_dir, kind, key, keep=None, dtype=None):
    """{feat_dir}/{kind}/{key}_*.pt 묶음 파일을 순서대로 읽는다. keep 이 있으면 그 이름의 클립만 남긴다.
    반환: (이름 목록, PackedTensors)"""
    files = sorted(glob.glob(os.path.join(feat_dir, kind, f'{key}_*.pt')))
    if not files:
        raise FileNotFoundError(f"특징이 없습니다: {os.path.join(feat_dir, kind, key + '_*.pt')}  (features 단계를 먼저 실행하세요)")
    names, parts = [], []
    for f in files:
        d = torch.load(f)
        n, x = d['names'], d['data']
        if keep is not None:
            idx = [i for i, nm in enumerate(n) if nm in keep]
            if not idx:
                continue
            if len(idx) != len(n):
                x = x[torch.tensor(idx)].clone()          # 필요한 클립만 복사(원래 묶음은 바로 해제)
                n = [n[i] for i in idx]
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype)
        names += n
        parts.append(x)
    return names, PackedTensors(parts)


class DataGenerator(Dataset):
    def __init__(self, params, feat_dir, split, with_labels=True, stages=(0,), keep=None):
        """
        Args:
            params (dict): config.py 가 만든 params.
            feat_dir (str): 데이터셋별 특징 캐시 폴더.
            split (str): 'train' | 'val' | 'test'.
            with_labels (bool): False 이면 (입력)만 돌려준다.
            stages: train 일 때 합칠 단계 번호들. 예: (0, 1) = S0 + S1. val/test 는 (0,)만 쓴다.
            keep: {단계: 이름 집합 또는 None} 필터. None 이면 그 단계의 클립을 전부 쓴다.
        """
        super().__init__()
        self.params = params
        self.split = split
        self.modality = params['modality']
        self.with_labels = with_labels
        self.stages = tuple(stages)
        self.stage_counts = {}
        a_dtype = torch.float16 if params.get('audio_dtype', 'float32') == 'float16' else None

        self.names, a_parts, v_parts, l_parts = [], [], [], []
        for s in self.stages:
            key = stage_key(split, s)
            kp = (keep or {}).get(s)
            n_a, audio = load_packed(feat_dir, 'audio', key, kp, a_dtype)
            if self.modality == 'audio_visual':
                n_v, video = load_packed(feat_dir, f"video_{params['video_crop']}", key, kp)
                assert n_v == n_a, f"오디오와 영상 특징의 클립 목록이 다릅니다 (단계 {s})"
                v_parts += video.parts
            if with_labels:
                n_l, labels = load_packed(feat_dir, 'labels_adpit' if params['multiACCDOA'] else 'labels_single', key, kp)
                assert n_l == n_a, f"오디오와 라벨 특징의 클립 목록이 다릅니다 (단계 {s})"
                l_parts += labels.parts
            a_parts += audio.parts
            self.names += n_a
            self.stage_counts[s] = len(n_a)
        out_dt = torch.float32 if a_dtype is not None else None
        self.audio = PackedTensors(a_parts, out_dt)
        self.video = PackedTensors(v_parts) if self.modality == 'audio_visual' else None
        self.labels = PackedTensors(l_parts) if with_labels else None

    def __getitem__(self, item):
        """
        Returns:
            audio_visual: ((audio_features, video_features), labels)   audio: (audio_features, labels)
            with_labels=False 이면 labels 없이 입력만.
        """
        audio_features = self.audio[item]
        video_features = self.video[item] if self.video is not None else None
        if self.with_labels:
            labels = self.labels[item]
            if not self.params['multiACCDOA']:
                mask = labels[:, :self.params['nb_classes']]
                mask = mask.repeat(1, 4)
                labels = mask * labels[:, self.params['nb_classes']:]
            if self.modality != 'audio_visual':
                # no need for on/off labels for audio only task
                if self.params['multiACCDOA']:
                    labels = labels[:, :, :-1, :]      # (L, 6, 5, 13) -> (L, 6, 4, 13)
                else:
                    labels = labels[:, :-self.params['nb_classes']]
        if self.modality == 'audio_visual':
            inputs = (audio_features, video_features)
        else:
            inputs = audio_features
        return (inputs, labels) if self.with_labels else inputs

    def __len__(self):
        return len(self.names)

    def ram_gb(self):
        """메모리에 올려 둔 특징의 대략적인 크기(GB)."""
        tot = 0
        for pk in (self.audio, self.video, self.labels):
            if pk is not None:
                tot += sum(p.numel() * p.element_size() for p in pk.parts)
        return tot / 2 ** 30
