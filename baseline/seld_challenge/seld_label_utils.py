# [변경 이력 시작]
#   2026-10-03  최초 생성: 원본 utils.py에서 torch 없이 쓰는 라벨 읽기와 각도 함수를 분리, 접지 않는 원형 각도 차이 추가
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
seld_label_utils.py

torch 없이 쓸 수 있는 라벨 읽기와 각도 계산 함수 모음.
원본 utils.py에서 가져와 고친 것이다(바뀐 곳은 CHANGES.md 참조). 채점(score.py, metrics.py)은 이 파일만 쓰므로
torch가 없는 PC에서도 돌아간다.
"""

import warnings
import numpy as np
from scipy import stats


def load_labels(label_file, convert_to_cartesian=True):
    """헤더가 있는 csv(frame,class,source,azimuth,distance,onscreen)를 읽는다. 헤더만 있는 파일은 {}를 돌려준다."""
    label_data = {}
    with open(label_file, 'r') as file:
        lines = file.readlines()[1:]  # Skip the header
        for line in lines:
            if not line.strip():
                continue
            values = line.strip().split(',')
            frame_idx = int(values[0])
            data_row = [int(values[1]), int(values[2]), float(values[3]), float(values[4]), int(values[5])]
            if frame_idx not in label_data:
                label_data[frame_idx] = []
            label_data[frame_idx].append(data_row)

    if convert_to_cartesian:
        label_data = convert_polar_to_cartesian(label_data)
    return label_data


def organize_labels(input_dict, max_frames, max_tracks=10):
    """
    :param input_dict: frame-index -> [[class-index, source-index, azimuth, distance, onscreen], ...]
    :param max_frames: 클립의 전체 프레임 수 (채점할 프레임 범위는 0 ~ max_frames-1)
    :return: dictionary_name[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
    """
    tracks = set(range(max_tracks))
    output_dict = {x: {} for x in range(max_frames)}
    for frame_idx in range(0, max_frames):
        if frame_idx not in input_dict:
            continue
        for [class_idx, source_idx, az, dist, onscreen] in input_dict[frame_idx]:
            if class_idx not in output_dict[frame_idx]:
                output_dict[frame_idx][class_idx] = {}
            if source_idx not in output_dict[frame_idx][class_idx] and source_idx < max_tracks:
                track_idx = source_idx  # If possible, use source_idx as track_idx
            else:                       # If not, use the first one available
                try:
                    track_idx = list(set(tracks) - output_dict[frame_idx][class_idx].keys())[0]
                except IndexError:
                    warnings.warn("The number of sources of is higher than the number of tracks. "
                                  "Some events will be missed.")
                    track_idx = 0  # Overwrite one event
            output_dict[frame_idx][class_idx][track_idx] = [az, dist, onscreen]

    return output_dict


def convert_polar_to_cartesian(input_dict):
    output_dict = {}
    for frame_idx in input_dict.keys():
        if frame_idx not in output_dict:
            output_dict[frame_idx] = []
        for tmp_val in input_dict[frame_idx]:
            azi_rad = tmp_val[2]*np.pi/180
            x = np.cos(azi_rad)
            y = np.sin(azi_rad)
            output_dict[frame_idx].append(tmp_val[0:2] + [x, y] + tmp_val[3:])
    return output_dict


def convert_cartesian_to_polar(input_dict):
    output_dict = {}
    for frame_idx in input_dict.keys():
        if frame_idx not in output_dict:
            output_dict[frame_idx] = []
        for tmp_val in input_dict[frame_idx]:
            x = tmp_val[2]
            y = tmp_val[3]
            azi_rad = np.arctan2(y, x)
            azimuth = azi_rad * 180 / np.pi
            output_dict[frame_idx].append(tmp_val[0:2] + [azimuth] + tmp_val[4:])
    return output_dict


def distance_between_cartesian_coordinates(x1, y1, x2, y2):
    """두 (x, y) 방향 사이의 각도(도)."""
    N1 = np.sqrt(x1**2 + y1**2 + 1e-10)
    N2 = np.sqrt(x2**2 + y2**2 + 1e-10)
    x1, y1, x2, y2 = x1/N1, y1/N1, x2/N2, y2/N2
    dist = x1*x2 + y1*y2
    dist = np.clip(dist, -1, 1)
    dist = np.arccos(dist) * 180 / np.pi
    return dist


def fold_az_angle(az):
    """방위각을 [-90, 90]으로 접는다(앞뒤 구분을 없앤다). DCASE 기본 방식."""
    az = np.asarray(az, dtype=float)
    az = (az + 180) % 360 - 180  # Make sure az is in the range [-180, 180)
    az_fold = az.copy()
    az_fold[np.logical_and(-180 <= az, az < -90)] = -180 - az[np.logical_and(-180 <= az, az < -90)]
    az_fold[np.logical_and(90 < az, az <= 180)] = 180 - az[np.logical_and(90 < az, az <= 180)]
    return az_fold


def circular_az_difference(az1, az2):
    """접지 않은 방위각의 차이 |((a - b + 180) mod 360) - 180|, 범위 0~180."""
    az1, az2 = np.asarray(az1, dtype=float), np.asarray(az2, dtype=float)
    return np.abs((az1 - az2 + 180) % 360 - 180)


def az_difference(az1, az2, fold):
    """fold=True: 둘 다 접은 뒤 차이(DCASE 기본 점수). fold=False: 원형 차이(확장 점수)."""
    if fold:
        return np.abs(fold_az_angle(az1) - fold_az_angle(az2))
    return circular_az_difference(az1, az2)


def least_distance_between_gt_pred(gt_list, pred_list, fold=False):
    """
    정답/예측 방위각 목록 사이의 최소 비용 짝짓기(헝가리안).
    fold=False(기본, 확장 점수): 접지 않은 원형 차이. fold=True(기본 점수): 접은 뒤 차이.
    :return: cost, row_ind, col_ind
    """
    from scipy.optimize import linear_sum_assignment
    gt_len, pred_len = gt_list.shape[0], pred_list.shape[0]
    ind_pairs = np.array([[x, y] for y in range(pred_len) for x in range(gt_len)])
    cost_mat = np.zeros((gt_len, pred_len))

    if gt_len and pred_len:
        az1, az2 = gt_list[ind_pairs[:, 0]], pred_list[ind_pairs[:, 1]]
        distances_ang = az_difference(az1, az2, fold)
        cost_mat[ind_pairs[:, 0], ind_pairs[:, 1]] = distances_ang

    row_ind, col_ind = linear_sum_assignment(cost_mat)
    cost = cost_mat[row_ind, col_ind]
    return cost, row_ind, col_ind


def jackknife_estimation(global_value, partial_estimates, significance_level=0.05):
    """원본 그대로(현재 채점에서는 쓰지 않는다)."""
    mean_jack_stat = np.mean(partial_estimates)
    n = len(partial_estimates)
    bias = (n - 1) * (mean_jack_stat - global_value)
    std_err = np.sqrt((n - 1) * np.mean((partial_estimates - mean_jack_stat) * (partial_estimates - mean_jack_stat), axis=0))
    estimate = global_value - bias
    if not (0 < significance_level < 1):
        raise ValueError("confidence level must be in (0, 1).")
    t_value = stats.t.ppf(1 - significance_level / 2, n - 1)
    conf_interval = estimate + t_value * np.array((-std_err, std_err))
    return estimate, bias, std_err, conf_interval
