# [변경 이력 시작]
#   2026-10-03  원본에서 복사 후 수정: load_video가 30fps를 가정하던 것을 실제 fps로 계산하고 프레임 수 검사 추가, 라벨 함수는 seld_label_utils로 분리, onscreen 확률을 int()로 잘라 저장하던 버그를 0.5 기준으로 수정, setup()과 print_results() 제거
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
utils.py

This module includes miscellaneous utility functions that support the project,
such as data_preprocessing, logging, file handling, and general-purpose helpers.

Author: Parthasaarathy Sudarsanam, Audio Research Group, Tampere University
Date: February 2025
"""

import os
import librosa
import librosa.feature
import numpy as np
import cv2
from PIL import Image
import torch
import warnings
from seld_label_utils import (load_labels, organize_labels, convert_polar_to_cartesian, convert_cartesian_to_polar,
                              distance_between_cartesian_coordinates, fold_az_angle, circular_az_difference,
                              least_distance_between_gt_pred, jackknife_estimation)  # torch 없이 쓰는 함수는 seld_label_utils.py로 옮김


def load_audio(audio_file, sampling_rate):
    """
    Loads an audio file.
    Args:
        audio_file (str): Path to the audio file.
        sampling_rate (int): Target sampling rate
    Returns:
        tuple: (audio_data, sample_rate)
    """
    audio_data, sr = librosa.load(path=audio_file, sr=sampling_rate, mono=False)
    return audio_data, sr


def extract_stft(audio, n_fft, hop_length, win_length):
    stft = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length).T
    return stft


def extract_log_mel_spectrogram(audio, sr, n_fft, hop_length, win_length, nb_mels):
    """
    Computes the log Mel spectrogram from an audio signal.

    Parameters:
        audio (ndarray): NumPy array containing the audio waveform.
        sr (int): The sample rate of the audio signal.
        n_fft (int): Size of the FFT window.
        hop_length (int): Number of samples to shift between successive frames.
        win_length (int): Length of each windowed frame in samples.
        nb_mels (int): Number of Mel filter banks to use.

    Returns:
        ndarray: Array of shape (2, time_frames, nb_mels) - log Mel spectrogram for each channel.
    """

    linear_stft = extract_stft(audio, n_fft, hop_length, win_length)
    linear_stft_mag = np.abs(linear_stft) ** 2
    mel_spec = librosa.feature.melspectrogram(S=linear_stft_mag, sr=sr, n_mels=nb_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spec)
    log_mel_spectrogram = log_mel_spectrogram.transpose((2, 0, 1))
    return log_mel_spectrogram


def load_video(video_file, fps, expected_frames=None, size=(360, 180)):
    """
    Loads video frames from a video file.
    Args:
        video_file (str): Path to the video file.
        fps (int): Target frames per second
        expected_frames (int): 라벨 프레임 수(200 또는 50). 영상 프레임 수가 다르면 에러를 낸다.
        size (tuple): (너비, 높이)로 줄일 크기. 원래 전처리는 (360, 180).
    Returns:
        list: List of PIL images (frames).
    """

    cap = cv2.VideoCapture(video_file)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        raise ValueError(f"fps를 읽을 수 없음: {video_file}")
    frame_interval = max(1, round(video_fps / fps))   # 원본은 fps를 30으로 고정하고 3장마다 1장을 썼다. 우리 영상(10fps)은 1이 된다.
    pil_frames = []
    frame_cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_cnt % frame_interval == 0:  # smoothening may not be required since we process the frames individually through a resnet
            resized_frame = cv2.resize(frame, size)
            frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
            pil_frames.append(pil_frame)
        frame_cnt += 1
    cap.release()
    if expected_frames is not None and len(pil_frames) != expected_frames:
        raise ValueError(f"영상 프레임 수 {len(pil_frames)} != 라벨 프레임 수 {expected_frames}: {video_file} (fps={video_fps})")
    return pil_frames


def extract_resnet_features(video_frames, resnet_preprocessor, resnet_backbone, device, batch=100):
    """
    Extracts ResNet-50 features from video frames.
    Args:
        video_frames (list): List of PIL video frames.
        resnet_preprocessor (callable): PIL 이미지 -> 전처리된 텐서.
        resnet_backbone (torch.nn.Module): Pre-trained ResNet model to extract features.
        device (str): Device to perform computation on (e.g., 'cuda' or 'cpu').
        batch (int): 한 번에 ResNet에 넣는 프레임 수.
    Returns:
        tensor: (nb_frames, 7, 7)  (video_crop=full 이면 (nb_frames, 7, 14))
    """

    with torch.no_grad():
        outs = []
        for i in range(0, len(video_frames), batch):
            imgs = torch.stack([resnet_preprocessor(image) for image in video_frames[i:i + batch]], dim=0).to(device)
            outs.append(torch.mean(resnet_backbone(imgs), dim=1).cpu())   # 채널 평균
        return torch.cat(outs, dim=0)


def process_labels(_desc_file, _nb_label_frames, _nb_unique_classes):

    se_label = torch.zeros((_nb_label_frames, _nb_unique_classes))
    x_label = torch.zeros((_nb_label_frames, _nb_unique_classes))
    y_label = torch.zeros((_nb_label_frames, _nb_unique_classes))
    dist_label = torch.zeros((_nb_label_frames, _nb_unique_classes))
    onscreen_label = torch.zeros((_nb_label_frames, _nb_unique_classes))

    for frame_ind, active_event_list in _desc_file.items():
        if frame_ind < _nb_label_frames:
            for active_event in active_event_list:
                se_label[frame_ind, active_event[0]] = 1
                x_label[frame_ind, active_event[0]] = active_event[2]
                y_label[frame_ind, active_event[0]] = active_event[3]
                dist_label[frame_ind, active_event[0]] = active_event[4] / 100.
                onscreen_label[frame_ind, active_event[0]] = active_event[5]

    label_mat = torch.cat((se_label, x_label, y_label, dist_label, onscreen_label), dim=1)
    return label_mat


def process_labels_adpit(_desc_file, _nb_label_frames, _nb_unique_classes):

    se_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))  # 50, 6, 13
    x_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))
    y_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))
    dist_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))
    onscreen_label = torch.zeros((_nb_label_frames, 6, _nb_unique_classes))

    for frame_ind, active_event_list in _desc_file.items():
        if frame_ind < _nb_label_frames:
            active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
            active_event_list_per_class = []
            for i, active_event in enumerate(active_event_list):
                active_event_list_per_class.append(active_event)
                if i == len(active_event_list) - 1:  # if the last
                    if len(active_event_list_per_class) == 1:  # if no ov from the same class
                        # a0----
                        active_event_a0 = active_event_list_per_class[0]
                        se_label[frame_ind, 0, active_event_a0[0]] = 1
                        x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                        y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                        dist_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4] / 100.
                        onscreen_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[5]
                    elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                        # --b0--
                        active_event_b0 = active_event_list_per_class[0]
                        se_label[frame_ind, 1, active_event_b0[0]] = 1
                        x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                        y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                        dist_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4] / 100.
                        onscreen_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[5]
                        # --b1--
                        active_event_b1 = active_event_list_per_class[1]
                        se_label[frame_ind, 2, active_event_b1[0]] = 1
                        x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                        y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                        dist_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4] / 100.
                        onscreen_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[5]

                    else:  # if ov with more than 2 sources from the same class
                        # ----c0
                        active_event_c0 = active_event_list_per_class[0]
                        se_label[frame_ind, 3, active_event_c0[0]] = 1
                        x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                        y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                        dist_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4] / 100.
                        onscreen_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[5]

                        # ----c1
                        active_event_c1 = active_event_list_per_class[1]
                        se_label[frame_ind, 4, active_event_c1[0]] = 1
                        x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                        y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                        dist_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4] / 100.
                        onscreen_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[5]
                        # ----c2
                        active_event_c2 = active_event_list_per_class[2]
                        se_label[frame_ind, 5, active_event_c2[0]] = 1
                        x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                        y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                        dist_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4] / 100.
                        onscreen_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[5]

                elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                    if len(active_event_list_per_class) == 1:  # if no ov from the same class
                        # a0----
                        active_event_a0 = active_event_list_per_class[0]
                        se_label[frame_ind, 0, active_event_a0[0]] = 1
                        x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                        y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                        dist_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4] / 100.
                        onscreen_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[5]
                    elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                        # --b0--
                        active_event_b0 = active_event_list_per_class[0]
                        se_label[frame_ind, 1, active_event_b0[0]] = 1
                        x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                        y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                        dist_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4] / 100.
                        onscreen_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[5]
                        # --b1--
                        active_event_b1 = active_event_list_per_class[1]
                        se_label[frame_ind, 2, active_event_b1[0]] = 1
                        x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                        y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                        dist_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4] / 100.
                        onscreen_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[5]
                    else:  # if ov with more than 2 sources from the same class
                        # ----c0
                        active_event_c0 = active_event_list_per_class[0]
                        se_label[frame_ind, 3, active_event_c0[0]] = 1
                        x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                        y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                        dist_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4] / 100.
                        onscreen_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[5]
                        # ----c1
                        active_event_c1 = active_event_list_per_class[1]
                        se_label[frame_ind, 4, active_event_c1[0]] = 1
                        x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                        y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                        dist_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4] / 100.
                        onscreen_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[5]
                        # ----c2
                        active_event_c2 = active_event_list_per_class[2]
                        se_label[frame_ind, 5, active_event_c2[0]] = 1
                        x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                        y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                        dist_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4] / 100.
                        onscreen_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[5]
                    active_event_list_per_class = []

    label_mat = torch.stack((se_label, x_label, y_label, dist_label, onscreen_label), dim=2)  # [nb_frames, 6, 5(act+XY+dist+onscreen), max_classes]
    return label_mat


def get_accdoa_labels(logits, nb_classes, modality):
    x, y = logits[:, :, :nb_classes], logits[:, :, nb_classes:2 * nb_classes]
    sed = torch.sqrt(x ** 2 + y ** 2) > 0.5
    distance = logits[:, :, 2 * nb_classes: 3 * nb_classes]
    distance[distance < 0.] = 0.
    if modality == 'audio_visual':
        on_screen = logits[:, :, 3 * nb_classes: 4 * nb_classes]
    else:
        on_screen = torch.zeros_like(distance)  # don't care for audio modality
    dummy_src_id = torch.zeros_like(distance)
    return sed, dummy_src_id, x, y, distance, on_screen


def get_multiaccdoa_labels(logits, nb_classes, modality):
    if modality == 'audio':
        x0, y0 = logits[:, :, :1*nb_classes], logits[:, :, 1*nb_classes:2*nb_classes]
        sed0 = torch.sqrt(x0**2 + y0**2) > 0.5
        dist0 = logits[:, :, 2*nb_classes:3*nb_classes]
        dist0[dist0 < 0.] = 0
        doa0 = logits[:, :, :2*nb_classes]
        dummy_src_id0 = torch.zeros_like(dist0)
        on_screen0 = torch.zeros_like(dist0)

        x1, y1 = logits[:, :, 3*nb_classes:4 * nb_classes], logits[:, :, 4 * nb_classes: 5 * nb_classes]
        sed1 = torch.sqrt(x1 ** 2 + y1 ** 2) > 0.5
        dist1 = logits[:, :, 5 * nb_classes:6 * nb_classes]
        dist1[dist1 < 0.] = 0
        doa1 = logits[:, :, 3*nb_classes:5 * nb_classes]
        dummy_src_id1 = torch.zeros_like(dist1)
        on_screen1 = torch.zeros_like(dist1)

        x2, y2 = logits[:, :, 6*nb_classes:7 * nb_classes], logits[:, :, 7 * nb_classes:8 * nb_classes]
        sed2 = torch.sqrt(x2 ** 2 + y2 ** 2) > 0.5
        dist2 = logits[:, :, 8 * nb_classes:9 * nb_classes]
        dist2[dist2 < 0.] = 0
        doa2 = logits[:, :, 6*nb_classes:8 * nb_classes]
        dummy_src_id2 = torch.zeros_like(dist2)
        on_screen2 = torch.zeros_like(dist2)

        return sed0, dummy_src_id0, doa0,  dist0, on_screen0, sed1, dummy_src_id1, doa1,  dist1, on_screen1, sed2, dummy_src_id2, doa2,  dist2, on_screen2

    else:
        x0, y0 = logits[:, :, :1 * nb_classes], logits[:, :, 1 * nb_classes:2 * nb_classes]
        sed0 = torch.sqrt(x0 ** 2 + y0 ** 2) > 0.5
        dist0 = logits[:, :, 2 * nb_classes:3 * nb_classes]
        dist0[dist0 < 0.] = 0
        doa0 = logits[:, :, :2 * nb_classes]
        dummy_src_id0 = torch.zeros_like(dist0)
        on_screen0 = logits[:, :, 3 * nb_classes:4 * nb_classes]

        x1, y1 = logits[:, :, 4 * nb_classes:5 * nb_classes], logits[:, :, 5 * nb_classes: 6 * nb_classes]
        sed1 = torch.sqrt(x1 ** 2 + y1 ** 2) > 0.5
        dist1 = logits[:, :, 6 * nb_classes:7 * nb_classes]
        dist1[dist1 < 0.] = 0
        doa1 = logits[:, :, 4 * nb_classes:6 * nb_classes]
        dummy_src_id1 = torch.zeros_like(dist1)
        on_screen1 = logits[:, :, 7 * nb_classes:8 * nb_classes]

        x2, y2 = logits[:, :, 8 * nb_classes:9 * nb_classes], logits[:, :, 9 * nb_classes:10 * nb_classes]
        sed2 = torch.sqrt(x2 ** 2 + y2 ** 2) > 0.5
        dist2 = logits[:, :, 10 * nb_classes:11 * nb_classes]
        dist2[dist2 < 0.] = 0
        doa2 = logits[:, :, 8 * nb_classes:10 * nb_classes]
        dummy_src_id2 = torch.zeros_like(dist2)
        on_screen2 = logits[:, :, 11 * nb_classes:12 * nb_classes]

        return sed0, dummy_src_id0, doa0, dist0, on_screen0, sed1, dummy_src_id1, doa1, dist1, on_screen1, sed2, dummy_src_id2, doa2, dist2, on_screen2


def get_output_dict_format_single_accdoa(sed, src_id, x, y, dist, onscreen, convert_to_polar=True):
    output_dict = {}
    for frame_cnt in range(sed.shape[0]):
        for class_cnt in range(sed.shape[1]):
            if sed[frame_cnt][class_cnt] > 0.5:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                output_dict[frame_cnt].append([class_cnt, src_id[frame_cnt][class_cnt], x[frame_cnt][class_cnt], y[frame_cnt][class_cnt], dist[frame_cnt][class_cnt], onscreen[frame_cnt][class_cnt]])

    if convert_to_polar:
        output_dict = convert_cartesian_to_polar(output_dict)
    return output_dict


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def get_output_dict_format_multi_accdoa(sed0, dummy_src_id0, doa0, dist0, on_screen0, sed1, dummy_src_id1, doa1, dist1, on_screen1, sed2, dummy_src_id2, doa2, dist2, on_screen2, thresh_unify, nb_classes, convert_to_polar=True):
    output_dict = {}
    for frame_cnt in range(sed0.shape[0]):
        for class_cnt in range(sed0.shape[1]):
            flag_0sim1 = determine_similar_location(sed0[frame_cnt][class_cnt], sed1[frame_cnt][class_cnt], doa0[frame_cnt], doa1[frame_cnt], class_cnt, thresh_unify, nb_classes)
            flag_1sim2 = determine_similar_location(sed1[frame_cnt][class_cnt], sed2[frame_cnt][class_cnt], doa1[frame_cnt], doa2[frame_cnt], class_cnt, thresh_unify, nb_classes)
            flag_2sim0 = determine_similar_location(sed2[frame_cnt][class_cnt], sed0[frame_cnt][class_cnt], doa2[frame_cnt], doa0[frame_cnt], class_cnt, thresh_unify, nb_classes)

            # unify or not unify according to flag
            if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                if sed0[frame_cnt][class_cnt] > 0.5:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt,
                                                   dummy_src_id0[frame_cnt][class_cnt],
                                                   doa0[frame_cnt][class_cnt],
                                                   doa0[frame_cnt][class_cnt + nb_classes],
                                                   dist0[frame_cnt][class_cnt],
                                                   on_screen0[frame_cnt][class_cnt]])

                if sed1[frame_cnt][class_cnt] > 0.5:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt,
                                                   dummy_src_id1[frame_cnt][class_cnt],
                                                   doa1[frame_cnt][class_cnt],
                                                   doa1[frame_cnt][class_cnt + nb_classes],
                                                   dist1[frame_cnt][class_cnt],
                                                   on_screen1[frame_cnt][class_cnt]])

                if sed2[frame_cnt][class_cnt] > 0.5:
                    if frame_cnt not in output_dict:
                        output_dict[frame_cnt] = []
                    output_dict[frame_cnt].append([class_cnt,
                                                   dummy_src_id2[frame_cnt][class_cnt],
                                                   doa2[frame_cnt][class_cnt],
                                                   doa2[frame_cnt][class_cnt + nb_classes],
                                                   dist2[frame_cnt][class_cnt],
                                                   on_screen2[frame_cnt][class_cnt]])

            elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                if flag_0sim1:
                    if sed2[frame_cnt][class_cnt] > 0.5:
                        output_dict[frame_cnt].append([class_cnt,
                                                       dummy_src_id2[frame_cnt][class_cnt],
                                                       doa2[frame_cnt][class_cnt],
                                                       doa2[frame_cnt][class_cnt + nb_classes],
                                                       dist2[frame_cnt][class_cnt],
                                                       on_screen2[frame_cnt][class_cnt]])

                    doa_pred_fc = (doa0[frame_cnt] + doa1[frame_cnt]) / 2
                    dist_pred_fc = (dist0[frame_cnt] + dist1[frame_cnt]) / 2
                    on_screen_pred_fc = on_screen0[frame_cnt]  # TODO: How to choose
                    dummy_src_id_pred_fc = dummy_src_id0[frame_cnt]
                    output_dict[frame_cnt].append(
                        [class_cnt, dummy_src_id_pred_fc[class_cnt], doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + nb_classes],dist_pred_fc[class_cnt], on_screen_pred_fc[class_cnt]])

                elif flag_1sim2:
                    if sed0[frame_cnt][class_cnt] > 0.5:
                        output_dict[frame_cnt].append([class_cnt,
                                                       dummy_src_id0[frame_cnt][class_cnt],
                                                       doa0[frame_cnt][class_cnt],
                                                       doa0[frame_cnt][class_cnt + nb_classes],
                                                       dist0[frame_cnt][class_cnt],
                                                       on_screen0[frame_cnt][class_cnt]])

                    doa_pred_fc = (doa1[frame_cnt] + doa2[frame_cnt]) / 2
                    dist_pred_fc = (dist1[frame_cnt] + dist2[frame_cnt]) / 2
                    on_screen_pred_fc = on_screen1[frame_cnt]  # TODO: How to choose
                    dummy_src_id_pred_fc = dummy_src_id1[frame_cnt]

                    output_dict[frame_cnt].append(
                        [class_cnt, dummy_src_id_pred_fc[class_cnt], doa_pred_fc[class_cnt],
                         doa_pred_fc[class_cnt + nb_classes], dist_pred_fc[class_cnt], on_screen_pred_fc[class_cnt]])

                elif flag_2sim0:
                    if sed1[frame_cnt][class_cnt] > 0.5:
                        output_dict[frame_cnt].append([class_cnt,
                                                       dummy_src_id1[frame_cnt][class_cnt],
                                                       doa1[frame_cnt][class_cnt],
                                                       doa1[frame_cnt][class_cnt + nb_classes],
                                                       dist1[frame_cnt][class_cnt],
                                                       on_screen1[frame_cnt][class_cnt]])

                    doa_pred_fc = (doa2[frame_cnt] + doa0[frame_cnt]) / 2
                    dist_pred_fc = (dist2[frame_cnt] + dist0[frame_cnt]) / 2
                    on_screen_pred_fc = on_screen2[frame_cnt]  # TODO: How to choose
                    dummy_src_id_pred_fc = dummy_src_id2[frame_cnt]

                    output_dict[frame_cnt].append(
                        [class_cnt, dummy_src_id_pred_fc[class_cnt], doa_pred_fc[class_cnt],
                         doa_pred_fc[class_cnt + nb_classes], dist_pred_fc[class_cnt], on_screen_pred_fc[class_cnt]])

            elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                doa_pred_fc = (doa0[frame_cnt] + doa1[frame_cnt] + doa2[frame_cnt]) / 3
                dist_pred_fc = (dist0[frame_cnt] + dist1[frame_cnt] + dist2[frame_cnt]) / 3

                dummy_src_id_pred_fc = dummy_src_id0[frame_cnt]
                on_screen_pred_fc = on_screen0[frame_cnt]  # TODO: How to do this?

                output_dict[frame_cnt].append(
                    [class_cnt, dummy_src_id_pred_fc[class_cnt], doa_pred_fc[class_cnt], doa_pred_fc[class_cnt + nb_classes], dist_pred_fc[class_cnt], on_screen_pred_fc[class_cnt]])

    if convert_to_polar:
        output_dict = convert_cartesian_to_polar(output_dict)
    return output_dict


def write_to_dcase_output_format(output_dict, output_dir, filename, split='', convert_dist_to_cm=True):
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    file_path = os.path.join(output_dir,split, filename)
    with open(file_path, 'w') as f:
        f.write('frame,class,source,azimuth,distance,onscreen\n')
        # Write data
        for frame_ind, values in output_dict.items():
            for value in values:
                azimuth_rounded = round(float(value[2]))
                dist_rounded = round(float(value[3]) * 100) if convert_dist_to_cm else round(float(value[3]))
                # 원본은 int(value[4])로 썼는데, onscreen은 시그모이드 확률(0~1)이라 소수점이 버려져 거의 항상 0이 되었다. 0.5 기준으로 바꾼다.
                onscreen = int(float(value[4]) > 0.5)
                f.write(f"{int(frame_ind)},{int(value[0])},{int(value[1])},{azimuth_rounded},{dist_rounded},{onscreen}\n")


def write_logits_to_dcase_format(logits, params, output_dir, filelist, split=''):
    if not params['multiACCDOA']:
        sed, dummy_src_id, x, y, dist, onscreen = get_accdoa_labels(logits, params['nb_classes'], params['modality'])
        for i in range(sed.size(0)):
            sed_i, dummy_src_id_i, x_i, y_i, dist_i, onscreen_i = sed[i].cpu().numpy(), dummy_src_id[i].cpu().numpy(), x[i].cpu().numpy(), y[i].cpu().numpy(), dist[i].cpu().numpy(), onscreen[i].cpu().numpy()
            output_dict = get_output_dict_format_single_accdoa(sed_i, dummy_src_id_i, x_i, y_i, dist_i, onscreen_i, convert_to_polar=True)
            write_to_dcase_output_format(output_dict, output_dir, os.path.splitext(os.path.basename(filelist[i]))[0] + '.csv', split)

    else:
        (sed0, dummy_src_id0, doa0, dist0, on_screen0,
         sed1, dummy_src_id1, doa1, dist1, on_screen1,
         sed2, dummy_src_id2, doa2, dist2, on_screen2) = get_multiaccdoa_labels(logits, params['nb_classes'], params['modality'])

        for i in range(sed0.size(0)):
            sed0_i, dummy_src_id0_i, doa0_i, dist0_i, on_screen0_i = sed0[i].cpu().numpy(), dummy_src_id0[i].cpu().numpy(), doa0[i].cpu().numpy(), dist0[i].cpu().numpy(), on_screen0[i].cpu().numpy()
            sed1_i, dummy_src_id1_i, doa1_i, dist1_i, on_screen1_i = sed1[i].cpu().numpy(), dummy_src_id1[i].cpu().numpy(), doa1[i].cpu().numpy(), dist1[i].cpu().numpy(), on_screen1[i].cpu().numpy()
            sed2_i, dummy_src_id2_i, doa2_i, dist2_i, on_screen2_i = sed2[i].cpu().numpy(), dummy_src_id2[i].cpu().numpy(), doa2[i].cpu().numpy(), dist2[i].cpu().numpy(), on_screen2[i].cpu().numpy()

            output_dict = get_output_dict_format_multi_accdoa(sed0_i, dummy_src_id0_i, doa0_i, dist0_i, on_screen0_i,
                                                              sed1_i, dummy_src_id1_i, doa1_i, dist1_i, on_screen1_i,
                                                              sed2_i, dummy_src_id2_i, doa2_i, dist2_i, on_screen2_i, params['thresh_unify'], params['nb_classes'], convert_to_polar=True)
            write_to_dcase_output_format(output_dict, output_dir, os.path.splitext(os.path.basename(filelist[i]))[0] + '.csv', split)
