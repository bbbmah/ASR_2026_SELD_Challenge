# [변경 이력 시작]
#   2026-10-13  최초 생성: NTU 원본을 묶음 파일 읽기와 train 통계 정규화로 다시 씀(finetune 제거)
#   2026-10-13  with_labels=False 추가(test 는 입력만)
# [변경 이력 끝]
"""
data_generator.py

캐시된 특징(extract_features.py 가 만든 묶음 파일)을 메모리에 올려서 학습/평가용 데이터를 돌려준다.
NTU 저장소의 data_generator.py 에서 바꾼 점
 - fold 이름 대신 split 이름('train', 'val')으로 묶음 파일을 찾는다.
 - 클립마다 .pt 를 읽는 대신 묶음을 한 번에 읽어 메모리에 둔다(합치지 않고 목록으로 들고 인덱스로 꺼낸다).
 - 정규화(train 통계: stats_train.npz)를 묶음을 읽을 때 적용한다. val 도 train 통계를 쓴다.
 - 원본에 있던 `params['finetune']` 는 정의된 적이 없어서 KeyError 가 나므로 뺐다(fold3 만 쓰던 미세조정 옵션).
 - 소리만 쓰므로 라벨의 on/off(onscreen) 채널은 원본처럼 뺀다.
"""

import os
import glob
import bisect
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class DataGenerator(Dataset):
    def __init__(self, params, split, transform=None, with_labels=True):
        """
        Args:
            params (dict): parameters.py 의 값 (feat_dir 가 채워져 있어야 한다).
            split (str): 'train' | 'val' | 'test'
            with_labels (bool): False 이면 입력만 돌려준다(test 는 정답이 없다).
            transform: 학습 때 특징에 적용할 증강(training_utils). 없으면 None.
        """
        super().__init__()
        self.params = params
        self.split = split
        self.transform = transform
        self.with_labels = with_labels
        feat_dir = params['feat_dir']

        st = np.load(os.path.join(feat_dir, 'stats_train.npz'))
        mean = torch.from_numpy(st['mean'])[None, :, None, :]       # (1, C, 1, F)
        std = torch.from_numpy(st['std'])[None, :, None, :]

        a_files = sorted(glob.glob(os.path.join(feat_dir, 'audio', f'{split}_*.pt')))
        l_files = sorted(glob.glob(os.path.join(feat_dir, 'labels_adpit', f'{split}_*.pt'))) if with_labels else a_files
        if not a_files or len(a_files) != len(l_files):
            raise FileNotFoundError(f"{split} 특징이 없거나 오디오({len(a_files)})와 라벨({len(l_files)}) 묶음 수가 다릅니다: {feat_dir}")
        self.names, self.audio, self.labels, self.offsets = [], [], [], [0]
        for fa, fl in zip(a_files, l_files):
            da = torch.load(fa)
            if with_labels:
                dl = torch.load(fl)
                assert da['names'] == dl['names'], f'오디오와 라벨의 클립 목록이 다릅니다: {fa}'
                self.labels.append(dl['data'])
            self.names += da['names']
            self.audio.append((da['data'] - mean) / std)
            self.offsets.append(self.offsets[-1] + len(da['names']))

    def __getitem__(self, item):
        """Returns: (audio_features (C, T, F), labels (L, 6, 4, 13))"""
        k = bisect.bisect_right(self.offsets, int(item)) - 1
        j = int(item) - self.offsets[k]
        audio_features = self.audio[k][j]
        if not self.with_labels:
            return audio_features
        labels = self.labels[k][j]
        if not self.params['multiACCDOA']:
            mask = labels[:, :self.params['nb_classes']]
            mask = mask.repeat(1, 4)
            labels = mask * labels[:, self.params['nb_classes']:]

        # no need for on/off labels for audio only task
        if self.params['multiACCDOA']:
            labels = labels[:, :, :-1, :]
        else:
            labels = labels[:, :-self.params['nb_classes']]

        if self.transform is not None:
            audio_features = self.transform(audio_features)
        return audio_features, labels

    def __len__(self):
        return len(self.names)

    def ram_gb(self):
        return sum(t.numel() * t.element_size() for t in self.audio + self.labels) / 2 ** 30
