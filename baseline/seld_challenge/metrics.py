# [변경 이력 시작]
#   2026-10-03  원본에서 복사 후 수정: 확장(접지 않음)과 기본(접음) 점수를 한 번에 계산, 채점 길이를 클립 길이로, 정답 폴더를 하위 폴더 없이 읽기, 정답 없는 클래스 평균 제외 옵션, 예측 csv 없는 클립은 빈 예측으로 채점
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
metrics.py

평가 지표. 원본(David Diaz-Guerra, Tampere University)에서 아래를 고쳤다. 자세한 것은 CHANGES.md 참조.
 - 방위각 오차를 접지 않는 원형 차이로 계산하는 옵션(fold_azimuth=False)과, 접는 방식(True) 둘 다 지원한다.
   ComputeSELDResults는 항상 두 방식('ext' = 접지 않음, 'basic' = 접음)을 한 번에 계산한다.
 - 정답 폴더를 하위 폴더 없이 csv가 바로 들어 있는 폴더로 읽는다.
 - 채점 프레임 수를 정답의 마지막 프레임이 아닌 클립 길이(clip_len)로 쓴다.
 - 정답이 하나도 없는 클래스를 평균 F에서 뺄 수 있다(exclude_absent_classes).
 - 예측 csv가 없는 클립은 '아무것도 예측하지 않음'으로 센다.
torch를 쓰지 않는다.
"""

import os
import warnings
import numpy as np
from seld_label_utils import least_distance_between_gt_pred, load_labels, organize_labels


class SELDMetrics(object):

    def __init__(self, doa_threshold=20, dist_threshold=np.inf, reldist_threshold=np.inf, req_onscreen=True,
                 nb_classes=13, average='macro', fold_azimuth=False, exclude_absent_classes=True):
        """
        :param doa_threshold: DOA error threshold for location sensitive detection.
        :param dist_threshold: Distance error threshold for location sensitive detection.
        :param reldist_threshold: Relative distance error threshold for location sensitive detection.
        :param req_onscreen: Require correct onscreen estimation for localization sensitive detection.
        :param nb_classes: Number of sound classes.
        :param average: Whether 'macro' or 'micro' aggregate the results
        :param fold_azimuth: True면 방위각을 앞쪽으로 접어서 비교(DCASE 기본 점수), False면 접지 않음(확장 점수).
        :param exclude_absent_classes: (macro) 정답이 하나도 없는 클래스를 평균 F에서 뺀다.
        """
        self._nb_classes = nb_classes

        # Variables for Location-sensitive detection performance
        self._TP = np.zeros(self._nb_classes)
        self._FP = np.zeros(self._nb_classes)
        self._FP_spatial = np.zeros(self._nb_classes)
        self._FN = np.zeros(self._nb_classes)

        self._Nref = np.zeros(self._nb_classes)

        self._ang_T = doa_threshold
        self._dist_T = dist_threshold
        self._reldist_T = reldist_threshold
        self._req_onscreen = req_onscreen
        self._fold = fold_azimuth
        self._exclude_absent = exclude_absent_classes

        self._S = 0
        self._D = 0
        self._I = 0

        # Variables for Class-sensitive localization performance
        self._total_AngE = np.zeros(self._nb_classes)
        self._total_DistE = np.zeros(self._nb_classes)
        self._total_RelDistE = np.zeros(self._nb_classes)
        self._total_OnscreenCorrect = np.zeros(self._nb_classes)

        self._DE_TP = np.zeros(self._nb_classes)
        self._DE_FP = np.zeros(self._nb_classes)
        self._DE_FN = np.zeros(self._nb_classes)

        assert average in ['macro', 'micro'], "Only 'micro' and 'macro' average are supported"
        self._average = average

    def compute_seld_scores(self):
        """
        :return: F score, angular error, distance error, relative distance error, onscreen accuracy, and classwise results
        """
        eps = np.finfo(float).eps
        classwise_results = []
        if self._average == 'micro':
            # Location-sensitive detection performance
            F = self._TP.sum() / (
                        eps + self._TP.sum() + self._FP_spatial.sum() + 0.5 * (self._FP.sum() + self._FN.sum()))

            # Class-sensitive localization performance
            AngE = self._total_AngE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.nan
            DistE = self._total_DistE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.nan
            RelDistE = self._total_RelDistE.sum() / float(self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.nan
            OnscreenAq = self._total_OnscreenCorrect.sum() / float(
                self._DE_TP.sum() + eps) if self._DE_TP.sum() else np.nan

        elif self._average == 'macro':
            # Location-sensitive detection performance
            F = self._TP / (eps + self._TP + self._FP_spatial + 0.5 * (self._FP + self._FN))
            if self._exclude_absent:
                F[self._Nref == 0] = np.nan      # 정답이 없는 클래스는 평균에서 뺀다

            # Class-sensitive localization performance
            AngE = self._total_AngE / (self._DE_TP + eps)
            AngE[self._DE_TP == 0] = np.nan
            DistE = self._total_DistE / (self._DE_TP + eps)
            DistE[self._DE_TP == 0] = np.nan
            RelDistE = self._total_RelDistE / (self._DE_TP + eps)
            RelDistE[self._DE_TP == 0] = np.nan
            OnscreenAq = self._total_OnscreenCorrect / (self._DE_TP + eps)
            OnscreenAq[self._DE_TP == 0] = np.nan

            classwise_results = np.array([F, AngE, DistE, RelDistE, OnscreenAq])
            F, AngE = np.nanmean(F) if not np.all(np.isnan(F)) else np.nan, np.nanmean(AngE)
            DistE, RelDistE = np.nanmean(DistE), np.nanmean(RelDistE)
            OnscreenAq = np.nanmean(OnscreenAq)

        else:
            raise NotImplementedError('Only micro and macro averaging are supported.')

        return F, AngE, DistE, RelDistE, OnscreenAq, classwise_results

    def update_seld_scores(self, pred, gt):
        """
        :param pred: pred[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
        :param gt: gt[frame-index][class-index][track-index] = [azimuth, distance, onscreen]
        pred와 gt는 같은 프레임 수(클립 길이)로 organize_labels 해야 한다.
        """
        eps = np.finfo(float).eps

        for frame_cnt in range(len(gt.keys())):
            loc_FN, loc_FP = 0, 0
            for class_cnt in range(self._nb_classes):
                # Counting the number of reference tracks for each class
                nb_gt_doas = len(gt[frame_cnt][class_cnt]) if class_cnt in gt[frame_cnt] else None
                nb_pred_doas = len(pred[frame_cnt][class_cnt]) if class_cnt in pred[frame_cnt] else None
                if nb_gt_doas is not None:
                    self._Nref[class_cnt] += nb_gt_doas
                if class_cnt in gt[frame_cnt] and class_cnt in pred[frame_cnt]:
                    # True positives
                    gt_values = np.array(list(gt[frame_cnt][class_cnt].values()))
                    gt_az, gt_dist, gt_onscreeen = gt_values[:, 0], gt_values[:, 1], gt_values[:, 2]
                    pred_values = np.array(list(pred[frame_cnt][class_cnt].values()))
                    pred_az, pred_dist, pred_onscreeen = pred_values[:, 0], pred_values[:, 1], pred_values[:, 2]

                    # Reference and predicted track matching
                    doa_err_list, row_inds, col_inds = least_distance_between_gt_pred(gt_az, pred_az, fold=self._fold)
                    dist_err_list = np.abs(gt_dist[row_inds] - pred_dist[col_inds])
                    rel_dist_err_list = dist_err_list / (gt_dist[row_inds] + eps)
                    onscreen_correct_list = (gt_onscreeen[row_inds] == pred_onscreeen[col_inds])

                    # https://dcase.community/challenge2022/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#evaluation
                    Pc = len(pred_az)
                    Rc = len(gt_az)
                    FNc = max(0, Rc - Pc)
                    FPcinf = max(0, Pc - Rc)
                    Kc = min(Pc, Rc)
                    TPc = Kc
                    Lc = np.sum(np.any((doa_err_list > self._ang_T,
                                        dist_err_list > self._dist_T,
                                        rel_dist_err_list > self._reldist_T,
                                        np.logical_and(np.logical_not(onscreen_correct_list), self._req_onscreen)),
                                       axis=0))
                    FPct = Lc
                    FPc = FPcinf + FPct
                    TPct = Kc - FPct
                    assert Pc == TPct + FPc
                    assert Rc == TPct + FPct + FNc

                    self._total_AngE[class_cnt] += doa_err_list.sum()
                    self._total_DistE[class_cnt] += dist_err_list.sum()
                    self._total_RelDistE[class_cnt] += rel_dist_err_list.sum()
                    self._total_OnscreenCorrect[class_cnt] += onscreen_correct_list.sum()

                    self._TP[class_cnt] += TPct
                    self._DE_TP[class_cnt] += TPc

                    self._FP[class_cnt] += FPcinf
                    self._DE_FP[class_cnt] += FPcinf
                    self._FP_spatial[class_cnt] += FPct
                    loc_FP += FPc

                    self._FN[class_cnt] += FNc
                    self._DE_FN[class_cnt] += FNc
                    loc_FN += FNc

                elif class_cnt in gt[frame_cnt] and class_cnt not in pred[frame_cnt]:
                    # False negative
                    loc_FN += nb_gt_doas
                    self._FN[class_cnt] += nb_gt_doas
                    self._DE_FN[class_cnt] += nb_gt_doas
                elif class_cnt not in gt[frame_cnt] and class_cnt in pred[frame_cnt]:
                    # False positive
                    loc_FP += nb_pred_doas
                    self._FP[class_cnt] += nb_pred_doas
                    self._DE_FP[class_cnt] += nb_pred_doas
                else:
                    # True negative
                    pass


def _num(x):
    """numpy 숫자를 json에 쓸 수 있는 float로 (nan -> None)."""
    x = float(x)
    return None if np.isnan(x) else x


class ComputeSELDResults(object):
    def __init__(self, params, ref_files_folder, clip_len, exclude=None):
        """
        예측 csv 폴더와 정답 csv 폴더로 SELD 점수를 계산한다. 항상 두 방식('ext', 'basic')으로 계산한다.

        :param params: lad_doa_thresh, lad_dist_thresh, lad_reldist_thresh, lad_req_onscreen, modality, average,
                       nb_classes, exclude_absent_classes를 가진 dict.
        :param ref_files_folder: 정답 csv가 바로 들어 있는 폴더(하위 폴더가 있으면 그 안의 csv도 모두 읽는다).
        :param clip_len: 클립의 라벨 프레임 수(main20/fix20 200, ctrl5 50). 이 길이만큼 채점한다.
        :param exclude: 채점에서 뺄 클립 이름(확장자 없이)의 집합.
        """
        self._doa_thresh = params['lad_doa_thresh']
        self._dist_thresh = params['lad_dist_thresh']
        self._reldist_thresh = params['lad_reldist_thresh']
        self._req_onscreen = params['lad_req_onscreen']
        self._modality = params['modality']
        self._exclude_absent = params.get('exclude_absent_classes', True)
        self._clip_len = int(clip_len)

        if params['modality'] == 'audio' and params['lad_req_onscreen']:
            warnings.warn("'lad_req_onscreen' is set to True, but 'modality' is 'audio'. "
                          "Onscreen estimation for detection metrics is not applicable to an audio-only model. "
                          "Resetting 'lad_req_onscreen' To False.")
            self._req_onscreen = False

        exclude = {e[:-4] if e.endswith('.csv') else e for e in (exclude or [])}
        self._ref_labels = {}
        for root, _, files in os.walk(ref_files_folder):
            for ref_file in sorted(files):
                if not ref_file.endswith('.csv') or ref_file[:-4] in exclude:
                    continue
                gt_dict = load_labels(os.path.join(root, ref_file), convert_to_cartesian=False)
                self._ref_labels[ref_file] = organize_labels(gt_dict, self._clip_len)
        if not self._ref_labels:
            raise FileNotFoundError(f"정답 csv가 없습니다: {ref_files_folder}")

        self._average = params['average']
        self._nb_classes = params['nb_classes']

    def _new_metrics(self, fold):
        return SELDMetrics(doa_threshold=self._doa_thresh, req_onscreen=self._req_onscreen,
                           dist_threshold=self._dist_thresh, reldist_threshold=self._reldist_thresh,
                           nb_classes=self._nb_classes, average=self._average, fold_azimuth=fold,
                           exclude_absent_classes=self._exclude_absent)

    def _pack(self, scores):
        F, AngE, DistE, RelDistE, OnscreenAq, cw = scores
        out = dict(F=_num(F), DOA_err=_num(AngE), dist_err=_num(DistE), rel_dist_err=_num(RelDistE),
                   onscreen_acc=_num(OnscreenAq) if self._modality == 'audio_visual' else None)
        if len(cw):
            out['classwise'] = {name: [_num(v) for v in cw[i]] for i, name in
                                enumerate(['F', 'DOA_err', 'dist_err', 'rel_dist_err', 'onscreen_acc'])}
            if self._modality != 'audio_visual':
                out['classwise']['onscreen_acc'] = None
        return out

    def get_SELD_Results(self, pred_files_path):
        """
        :return: {'ext': {...}, 'basic': {...}, 'n_clips': int, 'n_missing_pred': int}
          ext   = 방위각을 접지 않고 비교한 점수 (순위와 모델 고르기에 쓴다)
          basic = 정답과 예측을 둘 다 앞쪽으로 접어서 비교한 점수 (DCASE 방식)
        예측 csv가 없는 클립은 아무것도 예측하지 않은 것으로 센다.
        """
        ext, basic = self._new_metrics(False), self._new_metrics(True)
        n_missing = 0
        for ref_file, gt in self._ref_labels.items():
            pred_path = os.path.join(pred_files_path, ref_file)
            if os.path.exists(pred_path):
                pred_dict = load_labels(pred_path, convert_to_cartesian=False)
            else:
                pred_dict = {}
                n_missing += 1
            pred = organize_labels(pred_dict, self._clip_len)
            ext.update_seld_scores(pred, gt)
            basic.update_seld_scores(pred, gt)
        if n_missing:
            warnings.warn(f"예측 csv가 없는 클립이 {n_missing}개 있습니다(빈 예측으로 채점).")
        return dict(ext=self._pack(ext.compute_seld_scores()), basic=self._pack(basic.compute_seld_scores()),
                    n_clips=len(self._ref_labels), n_missing_pred=n_missing)
