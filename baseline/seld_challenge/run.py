# [변경 이력 시작]
#   2026-10-03  최초 생성: 모든 단계(prepare, features, train, eval, swaptest, predict, score, summary)를 한 파일로
#   2026-10-03  소규모 시험 결과 반영: 제외 목록 평가를 _excl 파일로 따로 저장하고 summary를 제외 여부별로 계산, 학습 클립이 배치보다 적을 때 에러, summary의 격차 계산
#   2026-10-07  추가 학습 데이터(S1) 연동: 단계별 zip 풀기, 필터 선택, updates로 학습 길이 결정, datainfo 단계, 점수 파일에 데이터 구성 기록, summary에 data_gaps.csv
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
"""
run.py  --  모든 실험을 이 파일 하나로 돌린다.

    python run.py <단계> --dataset main20 --preset B2 --set drive_root=/content/drive/MyDrive/... --set seed=2

단계: prepare | features | datainfo | train | eval | swaptest | predict | score | summary
설정은 configs/base.yaml 에 있고, --set key=value 로 어떤 값이든 덮어쓸 수 있다. 자세한 사용법은 README_baseline.md.
"""

import os
import sys
import csv
import json
import time
import shutil
import random
import zipfile
import argparse
import numpy as np
import yaml

import config as C
from score import score_dirs, one_line, read_exclude_list


# ====================================================================== 공통
def log(*a):
    print(*a, flush=True)


def blank(v):
    return v is None or (isinstance(v, str) and v.strip() == '')


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + '.tmp', 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(path + '.tmp', path)


def eval_tag(cfg):
    """평가 데이터셋이 학습 데이터셋과 다르면 파일 이름에 붙인다."""
    return '' if cfg.eval_dataset == cfg.dataset else f'_{cfg.eval_dataset}'


def excl_tag(cfg):
    """제외 목록을 쓴 채점 결과는 다른 파일로 저장한다(제외 없는 결과를 덮어쓰지 않도록)."""
    return '' if blank(cfg.exclude_list) else '_excl'


def data_meta(cfg):
    """점수 파일에 같이 남기는 학습 데이터 구성(요약에서 S0 와 S0+S1 을 짝짓는 데 쓴다)."""
    d = dict(data_tag=cfg.data_tag, std_name=cfg.std_name, train_stages=','.join(str(s) for s in cfg.train_stages),
             train_family=','.join(cfg.train_family), train_mirror=cfg.train_mirror, train_fraction=cfg.train_fraction,
             updates=cfg.updates)
    ts = os.path.join(cfg.run_dir, 'train_summary.json')
    if os.path.exists(ts):
        t = json.load(open(ts, encoding='utf8'))
        d.update(train_clips=t.get('train_clips'), total_updates=t.get('total_updates'), epochs_done=t.get('epochs_done'))
    return d


def params_for(cfg, dataset):
    """데이터셋의 라벨 프레임 수에 맞춘 params (모델 구조는 길이와 무관하다)."""
    p = dict(cfg.params)
    p['label_sequence_length'] = C.LABEL_LEN[dataset]
    return p


def exclusion(cfg, dataset):
    """채점에서 뺄 클립 이름 집합. ctrl5 는 부모(main20) 클립 이름을 조각 이름으로 바꾼다."""
    if blank(cfg.exclude_list):
        return set()
    path = cfg.exclude_list if os.path.isabs(cfg.exclude_list) else os.path.join(C.HERE, cfg.exclude_list)
    pm = None
    if dataset == 'ctrl5':
        for cand in (os.path.join(cfg.data_root, 'ctrl5', 'private', 'parent_map.csv'),
                     os.path.join(cfg.drive_root, 'dataset', 'ctrl5', 'private', 'parent_map.csv')):
            if os.path.exists(cand):
                pm = cand
                break
        if pm is None:
            raise FileNotFoundError("ctrl5 의 parent_map.csv 를 찾을 수 없습니다 "
                                    "(data_root/ctrl5/private/ 또는 drive_root/dataset/ctrl5/private/).")
    return read_exclude_list(path, pm)


# ====================================================================== prepare
def zip_path(cfg, dataset, split):
    if dataset == 'main20':
        name = {'train': 'train.zip', 'val': 'val.zip', 'test': 'test_inputs.zip', 'private': 'private_test.zip'}[split]
        return os.path.join(cfg.public_zip_dir, name)
    return os.path.join(cfg.internal_zip_dir, f'internal_{dataset}_{split}.zip')


def ensure_split(cfg, dataset, split):
    """zip 을 data_root 에 푼다. 이미 풀려 있으면 건너뛴다."""
    marker = os.path.join(cfg.data_root, '.done', f'{dataset}_{split}')
    if os.path.exists(marker):
        log(f'  [건너뜀] {dataset}/{split} 은 이미 풀려 있음')
        return
    zp = zip_path(cfg, dataset, split)
    if not os.path.exists(zp):
        raise FileNotFoundError(f"zip 을 찾을 수 없습니다: {zp}")
    t0 = time.time()
    log(f'  풀기: {zp}')
    with zipfile.ZipFile(zp) as z:
        z.extractall(os.path.join(cfg.data_root, dataset))
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, 'w').write(str(time.time()))
    log(f'  완료 ({time.time() - t0:.0f}초)')


def needed_splits(cfg):
    """{데이터셋: [split...]}  학습 데이터셋은 train/val/test, 다른 평가 데이터셋은 val/test."""
    need = {cfg.dataset: list(C.SPLITS[cfg.dataset])}
    if cfg.eval_dataset != cfg.dataset:
        need[cfg.eval_dataset] = ['val', 'test']
    return need


# ---------------------------------------------------------------------- 추가 학습 데이터 (단계 1 이상)
def extra_stages(cfg):
    return [s for s in cfg.train_stages if s > 0]


def stage_zips(cfg, dataset, stage):
    import glob
    return sorted(glob.glob(os.path.join(cfg.generated_dir, f's{stage}', f's{stage}_{dataset}_train_p*.zip')))


def ensure_stage(cfg, dataset, stage):
    """추가 데이터 묶음 zip 을 모두 data_root/{dataset}/ 에 푼다(train/ 폴더에 기존 클립과 이름이 겹치지 않게 들어간다)."""
    zips = stage_zips(cfg, dataset, stage)
    if not zips:
        raise FileNotFoundError(f"단계 {stage} 의 {dataset} zip 이 없습니다: {os.path.join(cfg.generated_dir, f's{stage}')} / s{stage}_{dataset}_train_p*.zip")
    t0 = time.time(); done = 0
    for i, zp in enumerate(zips):
        marker = os.path.join(cfg.data_root, '.done', f'{dataset}_s{stage}_{os.path.basename(zp)}')
        if os.path.exists(marker):
            continue
        with zipfile.ZipFile(zp) as z:
            z.extractall(os.path.join(cfg.data_root, dataset))
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        open(marker, 'w').write(str(time.time()))
        done += 1
        if done % 5 == 0 or i + 1 == len(zips):
            log(f'  단계 {stage} {dataset}: {i + 1}/{len(zips)} 풀었음 ({time.time() - t0:.0f}초)')
    log(f'  [단계 {stage}] {dataset} zip {len(zips)}개 (이번에 푼 것 {done}개)')


def load_stage_manifest(cfg, stage):
    """단계의 main20 창 매니페스트(창 = main20 클립). 병합본이 없으면 묶음별 파일을 합친다."""
    import glob
    import pandas as pd
    d = os.path.join(cfg.generated_dir, f's{stage}')
    merged = os.path.join(d, f's{stage}_main20_manifest.csv')
    if os.path.exists(merged):
        return pd.read_csv(merged)
    parts = sorted(glob.glob(os.path.join(d, f's{stage}_main20_manifest_p*.csv')))
    if not parts:
        raise FileNotFoundError(f"매니페스트가 없습니다: {merged}")
    return pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)


def stage_selection(cfg, dataset):
    """{단계: 쓸 클립 이름 집합 또는 None(전부)}. 단계 1 이상에 가족, 거울, 비율 필터를 적용한다.
    비율은 '창' 단위로 무작위(시드 고정)로 고르고, main20 과 ctrl5 가 같은 창을 쓴다."""
    sel = {}
    for s in cfg.train_stages:
        fam, mir, frac = cfg.train_family, cfg.train_mirror, cfg.train_fraction
        if s == 0 or not (fam or mir is not None or frac < 1):
            sel[s] = None
            continue
        m = load_stage_manifest(cfg, s)
        if fam:
            m = m[m['family'].isin(fam)]
        if mir is not None:
            m = m[m['mirror'] == mir]
        wins = sorted(m['clip'].astype(str))
        if frac < 1:
            k = max(1, int(round(len(wins) * frac)))
            wins = sorted(str(x) for x in np.random.default_rng(12345 + s).choice(wins, k, replace=False))
        sel[s] = set(wins) if dataset == 'main20' else {f'{w}_{j}' for w in wins for j in range(4)}
    return sel


def stage_prepare(cfg):
    for ds, splits in needed_splits(cfg).items():
        for sp in splits:
            ensure_split(cfg, ds, sp)
    for s in extra_stages(cfg):
        ensure_stage(cfg, cfg.dataset, s)


# ====================================================================== features
def stage_features(cfg):
    from extract_features import SELDFeatureExtractor
    for ds, splits in needed_splits(cfg).items():
        dirs = C.dataset_dirs(cfg, ds)
        ex = SELDFeatureExtractor(params_for(cfg, ds), dirs, C.feat_dir(cfg, ds))
        for sp in splits:
            if not os.path.isdir(dirs['audio'](sp)):
                raise FileNotFoundError(f"{dirs['audio'](sp)} 가 없습니다. prepare 를 먼저 실행하세요.")
            log(f'특징 추출: {ds}/{sp}  (modality={cfg.modality}, video_crop={cfg.video_crop})')
            ex.extract_features(sp)
    if extra_stages(cfg):
        dirs = C.dataset_dirs(cfg, cfg.dataset)
        ex = SELDFeatureExtractor(params_for(cfg, cfg.dataset), dirs, C.feat_dir(cfg, cfg.dataset))
        for s in extra_stages(cfg):                   # 선택 필터와 무관하게 단계의 클립 전부를 뽑아 둔다(필터는 학습 때 적용)
            log(f'특징 추출: {cfg.dataset}/train 단계 {s}  (modality={cfg.modality}, video_crop={cfg.video_crop})  <- 오래 걸립니다')
            ex.extract_features('train', stage=s)


# ====================================================================== 학습/평가 공통
def set_seed(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    import torch
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def make_loader(ds, bs, shuffle, drop_last, workers):
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=drop_last, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def split_batch(batch, modality, with_labels, device):
    """배치를 (오디오, 영상, 라벨)로 나눠 장치로 보낸다."""
    labels = None
    if with_labels:
        inputs, labels = batch
        labels = labels.to(device)
    else:
        inputs = batch
    if modality == 'audio_visual':
        return inputs[0].to(device), inputs[1].to(device), labels
    return inputs.to(device), None, labels


def apply_eval_video(ds, mode, seed):
    """평가 때만 영상 특징을 바꾼다. zeros: 0 으로. shuffle: 다른 클립의 영상(어떤 클립도 자기 영상을 받지 않음, 시드 고정)."""
    import torch
    from data_generator import PackedTensors
    if mode == 'real' or ds.video is None:
        return
    if mode == 'zeros':
        ds.video = ds.video.zeros_like()
    elif mode == 'shuffle':
        n = len(ds)
        if n < 2:
            raise ValueError("shuffle 은 클립이 2개 이상 필요합니다")
        order = np.random.default_rng(1000 + seed).permutation(n)       # 무작위 순서를 만들고
        src = np.empty(n, dtype=np.int64)
        src[order] = np.roll(order, -1)                                   # 순서상 다음 클립의 영상을 받는다 -> 자기 자신을 받는 클립이 없다
        assert (src != np.arange(n)).all()
        ds.video = PackedTensors.from_tensor(ds.video.to_tensor()[torch.from_numpy(src)])


def predict_split(model, ds, bs, workers, params, out_dir, device, loss_fn=None):
    """ds 전체를 예측해 out_dir 에 csv 로 쓴다. loss_fn 이 있으면 평균 손실도 돌려준다."""
    import torch
    import utils
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    loader = make_loader(ds, bs, False, False, workers)
    model.eval()
    total, nb = 0.0, 0
    with torch.no_grad():
        for j, batch in enumerate(loader):
            audio, video, labels = split_batch(batch, params['modality'], ds.with_labels, device)
            logits = model(audio, video)
            if loss_fn is not None:
                total += loss_fn(logits, labels).item(); nb += 1
            utils.write_logits_to_dcase_format(logits, params, out_dir, ds.names[j * bs:(j + 1) * bs])
    return total / max(1, nb)


def build_model(cfg, device):
    from model import SELDModel
    return SELDModel(params=cfg.params).to(device)


def build_loss(cfg, device):
    from loss import SELDLossADPIT, SELDLossSingleACCDOA
    p = cfg.params
    return (SELDLossADPIT(params=p) if p['multiACCDOA'] else SELDLossSingleACCDOA(params=p)).to(device)


def score_val(cfg, pred_dir, dataset, exclude=None):
    ref = C.dataset_dirs(cfg, dataset)['labels']('val')
    return score_dirs(pred_dir, ref, C.LABEL_LEN[dataset], cfg.modality, exclude,
                      exclude_absent_classes=bool(cfg.exclude_absent_classes))


def metric_value(res, name):
    v = res['ext' if name == 'F_ext' else 'basic']['F']
    return float('-inf') if v is None else v


# ====================================================================== train
def stage_train(cfg):
    import torch
    from data_generator import DataGenerator
    device = get_device()
    set_seed(cfg.seed)
    p = cfg.params
    feat = C.feat_dir(cfg, cfg.dataset)
    os.makedirs(cfg.run_dir, exist_ok=True)

    cfg_dump = {k: v for k, v in cfg.items() if k != 'params'}
    cfg_dump['params'] = {k: (str(v) if isinstance(v, float) and v == float('inf') else v) for k, v in p.items()}
    yaml.safe_dump(cfg_dump, open(os.path.join(cfg.run_dir, 'config_used.yaml'), 'w', encoding='utf8'), allow_unicode=True, sort_keys=False)

    log(f'== 학습: {cfg.run_name}  (modality={cfg.modality}, video_input={cfg.video_input}, video_crop={cfg.video_crop}, '
        f'av_time_window={cfg.av_time_window}, batch={cfg.batch_size_train}, device={device})')
    keep = stage_selection(cfg, cfg.dataset)
    train_ds = DataGenerator(p, feat, 'train', stages=cfg.train_stages, keep=keep)
    val_ds = DataGenerator(p, feat, 'val')
    for s, kp in keep.items():                                 # 필터로 고른 클립이 특징에 없으면 알린다(features 가 덜 끝난 경우)
        if kp is not None and train_ds.stage_counts.get(s, 0) != len(kp):
            log(f'   [경고] 단계 {s}: 고른 클립 {len(kp)}개 중 특징이 있는 것은 {train_ds.stage_counts.get(s, 0)}개 (피크로 빠진 창이거나 features 가 덜 끝남)')
    train_loader = make_loader(train_ds, cfg.batch_size_train, True, True, p['nb_workers'])
    if len(train_loader) == 0:
        raise ValueError(f"train 클립 수({len(train_ds)})가 배치 크기({cfg.batch_size_train})보다 작습니다. batch_size 를 줄이세요.")
    steps = len(train_loader)
    if cfg.updates is not None:                                # 총 갱신 횟수로 에폭 수를 정한다
        p['nb_epochs'] = max(1, -(-cfg.updates // steps))
    log(f'   train {len(train_ds)}개 (단계별 {train_ds.stage_counts}), val {len(val_ds)}개, 에폭당 {steps}번 갱신 -> '
        f"{p['nb_epochs']}에폭 = 약 {p['nb_epochs'] * steps}번 갱신, 메모리 약 {train_ds.ram_gb():.1f}GB")
    cfg_dump['effective'] = dict(train_clips=len(train_ds), stage_counts={str(k): v for k, v in train_ds.stage_counts.items()},
                                 steps_per_epoch=steps, epochs=p['nb_epochs'], total_updates=p['nb_epochs'] * steps)
    yaml.safe_dump(cfg_dump, open(os.path.join(cfg.run_dir, 'config_used.yaml'), 'w', encoding='utf8'), allow_unicode=True, sort_keys=False)

    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=p['learning_rate'], weight_decay=p['weight_decay'])
    loss_fn = build_loss(cfg, device)

    start_epoch, best_val, best_epoch = 0, float('-inf'), -1
    last_path = os.path.join(cfg.run_dir, 'last.pth')
    best_path = os.path.join(cfg.run_dir, 'best_model.pth')
    log_path = os.path.join(cfg.run_dir, 'train_log.csv')
    if cfg.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck['seld_model']); optimizer.load_state_dict(ck['opt'])
        start_epoch, best_val, best_epoch = ck['epoch'] + 1, ck['best_value'], ck['best_epoch']
        log(f'   이어 하기: 에폭 {start_epoch + 1} 부터 (지금까지 최고 {cfg.select_metric}={best_val:.4f}, 에폭 {best_epoch + 1})')
    elif os.path.exists(log_path):
        os.remove(log_path)

    tmp_pred = os.path.join(cfg.local_tmp, cfg.run_name, 'pred_val')
    cols = ['epoch', 'train_loss', 'val_loss', 'F_ext', 'F_basic', 'DOA_ext', 'DOA_basic', 'rel_dist_ext', 'onscreen_ext', 'seconds']
    for epoch in range(start_epoch, p['nb_epochs']):
        t0 = time.time()
        model.train()
        tot = 0.0
        for batch in train_loader:
            audio, video, labels = split_batch(batch, cfg.modality, True, device)
            optimizer.zero_grad()
            loss = loss_fn(model(audio, video), labels)
            loss.backward()
            optimizer.step()
            tot += loss.item()
        train_loss = tot / len(train_loader)

        val_loss = predict_split(model, val_ds, cfg.batch_size_train, p['nb_workers'], p, tmp_pred, device, loss_fn)
        res = score_val(cfg, tmp_pred, cfg.dataset)
        cur = metric_value(res, cfg.select_metric)
        e, b = res['ext'], res['basic']
        row = dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss, F_ext=e['F'], F_basic=b['F'], DOA_ext=e['DOA_err'],
                   DOA_basic=b['DOA_err'], rel_dist_ext=e['rel_dist_err'], onscreen_ext=e['onscreen_acc'], seconds=time.time() - t0)
        new = not os.path.exists(log_path)
        with open(log_path, 'a', newline='', encoding='utf8') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new: w.writeheader()
            w.writerow(row)
        f_fmt = lambda v: 'NA' if v is None else f'{100 * v:.2f}'
        log(f"에폭 {epoch + 1}/{p['nb_epochs']} | 학습손실 {train_loss:.3f} | val손실 {val_loss:.3f} | "
            f"F_ext {f_fmt(e['F'])} F_basic {f_fmt(b['F'])} | DOA_ext {e['DOA_err'] if e['DOA_err'] is None else round(e['DOA_err'], 1)} | {row['seconds']:.0f}초")

        if cur >= best_val:
            best_val, best_epoch = cur, epoch
            torch.save({'seld_model': model.state_dict(), 'epoch': epoch, 'best_value': best_val, 'select_metric': cfg.select_metric,
                        'val_scores': res}, best_path + '.tmp')
            os.replace(best_path + '.tmp', best_path)
        torch.save({'seld_model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch,
                    'best_value': best_val, 'best_epoch': best_epoch}, last_path + '.tmp')
        os.replace(last_path + '.tmp', last_path)

    atomic_json(os.path.join(cfg.run_dir, 'train_summary.json'),
                dict(run=cfg.run_name, best_epoch=best_epoch + 1, best_value=best_val, select_metric=cfg.select_metric,
                     epochs_done=p['nb_epochs'], steps_per_epoch=steps, total_updates=p['nb_epochs'] * steps,
                     train_clips=len(train_ds), stage_counts={str(k): v for k, v in train_ds.stage_counts.items()}))
    log(f'== 학습 끝. 최고 {cfg.select_metric}={best_val:.4f} (에폭 {best_epoch + 1}). 다음: eval')


# ====================================================================== eval / swaptest
def load_best(cfg, device):
    import torch
    path = os.path.join(cfg.run_dir, 'best_model.pth')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 가 없습니다. train 을 먼저 실행하세요.")
    model = build_model(cfg, device)
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck['seld_model'])
    return model, ck


def eval_val_modes(cfg, modes):
    from data_generator import DataGenerator
    device = get_device()
    set_seed(cfg.seed)
    p = params_for(cfg, cfg.eval_dataset)
    model, ck = load_best(cfg, device)
    exclude = exclusion(cfg, cfg.eval_dataset)
    for mode in modes:
        ds = DataGenerator(p, C.feat_dir(cfg, cfg.eval_dataset), 'val')
        apply_eval_video(ds, mode, cfg.seed)
        out = os.path.join(cfg.run_dir, f'pred_val{eval_tag(cfg)}_{mode}')
        predict_split(model, ds, cfg.batch_size_eval, p['nb_workers'], p, out, device)
        res = score_val(cfg, out, cfg.eval_dataset, exclude)
        meta = dict(run=cfg.run_name, dataset=cfg.dataset, eval_dataset=cfg.eval_dataset, split='val', eval_video=mode, exclude_list=cfg.exclude_list or '',
                    preset=cfg.preset, seed=cfg.seed, modality=cfg.modality, video_input=cfg.video_input,
                    video_crop=cfg.video_crop, av_time_window=cfg.av_time_window, best_epoch=ck['epoch'] + 1,
                    excluded_clips=len(exclude), scores=res, **data_meta(cfg))
        atomic_json(os.path.join(cfg.run_dir, f'scores_val{eval_tag(cfg)}_{mode}{excl_tag(cfg)}.json'), meta)
        log(f'[val/{cfg.eval_dataset}/eval_video={mode}] ' + one_line(res))


def stage_eval(cfg):
    eval_val_modes(cfg, [cfg.eval_video])


def stage_swaptest(cfg):
    if cfg.modality != 'audio_visual':
        raise ValueError("swaptest 는 영상+소리 모델(audio_visual)에서만 동작합니다.")
    eval_val_modes(cfg, ['real', 'shuffle', 'zeros'])


# ====================================================================== predict / score
def stage_predict(cfg):
    from data_generator import DataGenerator
    device = get_device()
    set_seed(cfg.seed)
    p = params_for(cfg, cfg.eval_dataset)
    model, _ = load_best(cfg, device)
    ds = DataGenerator(p, C.feat_dir(cfg, cfg.eval_dataset), 'test', with_labels=False)
    apply_eval_video(ds, cfg.eval_video, cfg.seed)
    out = os.path.join(cfg.run_dir, f'pred_test{eval_tag(cfg)}')
    predict_split(model, ds, cfg.batch_size_eval, p['nb_workers'], p, out, device)
    zpath = os.path.join(cfg.run_dir, f'submission_{cfg.run_name}{eval_tag(cfg)}.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out)):
            z.write(os.path.join(out, f), f)
    log(f'test 예측 {len(ds)}개 -> {out}\n제출용 zip: {zpath}')


def stage_score(cfg):
    """주최자용: private 정답으로 test 예측을 채점한다."""
    ds = cfg.eval_dataset
    ref = C.dataset_dirs(cfg, ds)['test_labels']
    if not os.path.isdir(ref):
        ensure_split(cfg, ds, 'private')
    pred = os.path.join(cfg.run_dir, f'pred_test{eval_tag(cfg)}')
    exclude = exclusion(cfg, ds)
    res = score_dirs(pred, ref, C.LABEL_LEN[ds], cfg.modality, exclude, exclude_absent_classes=bool(cfg.exclude_absent_classes))
    meta = dict(run=cfg.run_name, dataset=cfg.dataset, eval_dataset=ds, split='test', eval_video=cfg.eval_video, exclude_list=cfg.exclude_list or '', preset=cfg.preset,
                seed=cfg.seed, modality=cfg.modality, video_input=cfg.video_input, video_crop=cfg.video_crop,
                av_time_window=cfg.av_time_window, excluded_clips=len(exclude), scores=res, **data_meta(cfg))
    atomic_json(os.path.join(cfg.run_dir, f'scores_test{eval_tag(cfg)}_{cfg.eval_video}{excl_tag(cfg)}.json'), meta)
    log('[test] ' + one_line(res))


# ====================================================================== summary
def flatten(meta):
    row = {k: meta.get(k) for k in ('run', 'dataset', 'eval_dataset', 'split', 'eval_video', 'preset', 'seed', 'modality',
                                    'video_input', 'video_crop', 'av_time_window', 'best_epoch', 'exclude_list', 'excluded_clips',
                                    'train_family', 'train_mirror', 'train_fraction', 'updates', 'train_clips', 'total_updates', 'epochs_done')}
    row['data_tag'] = meta.get('data_tag') or ''                      # 옛 결과(S0)에는 없으므로 빈 문자열
    row['train_stages'] = meta.get('train_stages', '0')
    row['std_name'] = meta.get('std_name') or f"{meta.get('dataset')}_{meta.get('preset')}_s{meta.get('seed')}"
    s = meta['scores']
    for kind in ('ext', 'basic'):
        for key, name in (('F', 'F'), ('DOA_err', 'DOA'), ('rel_dist_err', 'relDist'), ('dist_err', 'dist'), ('onscreen_acc', 'onscreen')):
            row[f'{name}_{kind}'] = s[kind][key]
    row['n_clips'] = s['n_clips']
    return row


def stage_summary(cfg):
    import pandas as pd
    runs_dir = os.path.join(cfg.out_root, 'runs')
    rows = []
    for d in sorted(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else []:
        for f in sorted(os.listdir(os.path.join(runs_dir, d))):
            if f.startswith('scores_') and f.endswith('.json'):
                rows.append(flatten(json.load(open(os.path.join(runs_dir, d, f), encoding='utf8'))))
    if not rows:
        log('채점 결과(scores_*.json)가 아직 없습니다.')
        return
    df = pd.DataFrame(rows)
    df['exclude_list'] = df['exclude_list'].fillna('')
    os.makedirs(cfg.results_dir, exist_ok=True)
    df.to_csv(os.path.join(cfg.results_dir, 'summary.csv'), index=False)
    log(f'summary.csv: {len(df)}행 -> {cfg.results_dir}')

    # 영상 덕분에 오른 점수 = (영상+소리 모델) - (소리 모델). 표준 프리셋 이름(접미사 없음)이고 val/real 인 실행만 쓴다.
    # data_tag 는 학습 데이터 구성(빈 문자열 = 기존 S0). 격차는 같은 데이터 구성끼리만 계산한다.
    std = df[(df.split == 'val') & (df.eval_video == 'real') & (df.eval_dataset == df.dataset) & (df.run == df.std_name)]
    gaps_def = [('B2-B1', 'B2', 'B1'), ('B2-(B2-0)', 'B2', 'B2-0'), ('B3-B1', 'B3', 'B1')]
    g = []
    for (ds, tag, seed, ex), sub in std.groupby(['dataset', 'data_tag', 'seed', 'exclude_list']):
        by = {r.preset: r for r in sub.itertuples()}
        for name, a, b in gaps_def:
            if a in by and b in by:
                g.append(dict(dataset=ds, data_tag=tag, seed=seed, exclude_list=ex, gap=name, F_ext_gap=by[a].F_ext - by[b].F_ext, F_basic_gap=by[a].F_basic - by[b].F_basic))
    gd = pd.DataFrame(g)
    out = [gd] if len(gd) else []
    if len(gd):
        agg = gd.groupby(['dataset', 'data_tag', 'exclude_list', 'gap'])[['F_ext_gap', 'F_basic_gap']].agg(['mean', 'std'])
        agg.columns = [f'{a}_{b}' for a, b in agg.columns]
        agg = agg.reset_index().assign(seed='mean/std')
        out.append(agg)
        # main20 의 격차 - ctrl5 의 격차 (같은 데이터 구성, 같은 시드끼리)
        pv = gd.pivot_table(index=['data_tag', 'seed', 'exclude_list', 'gap'], columns='dataset', values=['F_ext_gap', 'F_basic_gap'])
        if ('F_ext_gap', 'main20') in pv.columns and ('F_ext_gap', 'ctrl5') in pv.columns:
            d = pd.DataFrame({'F_ext_gap': pv[('F_ext_gap', 'main20')] - pv[('F_ext_gap', 'ctrl5')],
                              'F_basic_gap': pv[('F_basic_gap', 'main20')] - pv[('F_basic_gap', 'ctrl5')]}).dropna().reset_index()
            d['dataset'] = 'main20-ctrl5'
            out.append(d)
            m = d.groupby(['data_tag', 'exclude_list', 'gap'])[['F_ext_gap', 'F_basic_gap']].agg(['mean', 'std'])
            m.columns = [f'{a}_{b}' for a, b in m.columns]
            out.append(m.reset_index().assign(dataset='main20-ctrl5', seed='mean/std'))
        gaps = pd.concat(out, ignore_index=True)
        gaps.to_csv(os.path.join(cfg.results_dir, 'gaps.csv'), index=False)
        log(f'gaps.csv -> {cfg.results_dir}')
        with pd.option_context('display.width', 200, 'display.max_columns', 20):
            log(gaps.to_string(index=False))
    else:
        log('gaps: 같은 데이터셋·시드에서 B1/B2/B2-0/B3 중 필요한 쌍의 val(real) 결과가 아직 없습니다.')

    # 추가 데이터의 효과: 같은 (데이터셋, 프리셋, 시드)에서 [기존 S0 실행] 대비 [데이터 구성이 다른 실행]의 val(real) 점수 차이
    base = std[std.data_tag == ''].set_index(['dataset', 'preset', 'seed', 'exclude_list'])
    dg = []
    for r in std[std.data_tag != ''].itertuples():
        k = (r.dataset, r.preset, r.seed, r.exclude_list)
        if k not in base.index:
            continue
        b = base.loc[k]
        dg.append(dict(dataset=r.dataset, preset=r.preset, seed=r.seed, data_tag=r.data_tag, train_clips=r.train_clips, total_updates=r.total_updates,
                       F_ext_S0=b.F_ext, F_ext_new=r.F_ext, F_ext_delta=r.F_ext - b.F_ext,
                       F_basic_S0=b.F_basic, F_basic_new=r.F_basic, F_basic_delta=r.F_basic - b.F_basic,
                       DOA_ext_delta=r.DOA_ext - b.DOA_ext, DOA_basic_delta=r.DOA_basic - b.DOA_basic,
                       onscreen_delta=(r.onscreen_ext - b.onscreen_ext) if r.onscreen_ext == r.onscreen_ext and b.onscreen_ext == b.onscreen_ext else None))
    if dg:
        dgd = pd.DataFrame(dg).sort_values(['dataset', 'preset', 'seed', 'data_tag'])
        dgd.to_csv(os.path.join(cfg.results_dir, 'data_gaps.csv'), index=False)
        log(f'data_gaps.csv (S0 대비 추가 데이터 효과) -> {cfg.results_dir}')
        with pd.option_context('display.width', 220, 'display.max_columns', 20):
            log(dgd.round(4).to_string(index=False))


# ====================================================================== datainfo
S0_CLIPS = {'main20': 2999, 'ctrl5': 11996}            # 기존 학습 데이터(S0)의 클립 수 (피크로 빠진 창 반영)


def stage_datainfo(cfg):
    """학습 데이터 구성을 미리 보여 준다. 매니페스트만 읽으므로 zip 풀기, 특징 추출 전에도 되고 torch 가 필요 없다."""
    import pandas as pd
    ds = cfg.dataset; L = C.LABEL_LEN[ds]
    a_b = 2 * (L * 5 + 1) * 64 * (2 if cfg.audio_dtype == 'float16' else 4)           # 오디오 특징(로그멜) 바이트
    l_b = L * 6 * 5 * 13 * 4                                                          # ADPIT 라벨
    v_b = (L * (49 if cfg.video_crop == 'center' else 98) * 4) if cfg.modality == 'audio_visual' else 0
    per = a_b + l_b + v_b
    clip_mb = {'main20': 3.56, 'ctrl5': 0.635}[ds]
    n_s0 = S0_CLIPS[ds]
    lab_dir = C.dataset_dirs(cfg, ds)['labels']('train')
    if os.path.isdir(lab_dir):                                   # 데이터를 이미 풀었으면 실제 개수를 센다 (기존 S0 = train_숫자.csv)
        import re
        n_real = sum(1 for f in os.listdir(lab_dir) if re.match(r'^train_\d+\.csv$', f))
        n_s0 = n_real if n_real else n_s0
    rows, total = [], 0
    sel = stage_selection(cfg, ds)
    for s in cfg.train_stages:
        if s == 0:
            n = n_s0
            rows.append((0, n, 'F1(기존 정의)' if ds == 'main20' else '고정 카메라', '거울 없음'))
        else:
            m = load_stage_manifest(cfg, s)
            if sel[s] is not None:
                keep = {w for w in m['clip'].astype(str) if (w in sel[s] if ds == 'main20' else all(f'{w}_{j}' in sel[s] for j in range(4)))}
                m = m[m['clip'].astype(str).isin(keep)]
            n = len(m) * (1 if ds == 'main20' else 4)
            fam = ', '.join(f'{k} {v}' for k, v in m['family'].value_counts().sort_index().items()) + ('' if ds == 'main20' else ' (창 수; 조각은 창당 4개)')
            rows.append((s, n, fam, f"거울 {int(m['mirror'].sum())}/{len(m)} ({m['mirror'].mean():.0%})"))
        total += n
    log(f'== 학습 데이터 구성: {cfg.run_name}  (dataset={ds}, 데이터 꼬리표="{cfg.data_tag or "(기존 S0)"}")')
    for s, n, fam, mir in rows:
        log(f'   단계 {s}: {n:6d}개  | 가족 {fam}  | {mir}')
    steps = total // cfg.batch_size_train
    log(f'   합계 {total}개, 배치 {cfg.batch_size_train} -> 에폭당 {steps}번 갱신')
    if cfg.updates is not None:
        ep = max(1, -(-cfg.updates // max(1, steps)))
        log(f'   updates={cfg.updates} -> {ep}에폭 (약 {ep * steps}번 갱신)')
    else:
        log(f'   epochs={cfg.epochs} -> 약 {cfg.epochs * steps}번 갱신  (참고: 기존 S0 학습은 약 {n_s0 // cfg.batch_size_train * 200}번)')
    log(f'   학습 특징 메모리 약 {total * per / 2**30:.1f}GB (오디오 {total * a_b / 2**30:.1f} + 라벨 {total * l_b / 2**30:.1f} + 영상 {total * v_b / 2**30:.1f}), '
        f'풀어 둔 클립 디스크 약 {total * clip_mb / 1024:.1f}GB')
    if total * per / 2 ** 30 > 9 and cfg.audio_dtype != 'float16':
        log('   [참고] 메모리가 9GB 를 넘습니다. 일반 램(12GB) 런타임이면 audio_dtype=float16 또는 고용량 램을 쓰세요.')


# ====================================================================== main
STAGE_FUNCS = dict(prepare=stage_prepare, features=stage_features, datainfo=stage_datainfo, train=stage_train, eval=stage_eval,
                   swaptest=stage_swaptest, predict=stage_predict, score=stage_score, summary=stage_summary)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage', choices=list(STAGE_FUNCS))
    ap.add_argument('--dataset', default=None, help='main20 | ctrl5')
    ap.add_argument('--preset', default=None, help='B1 | B2 | B2-0 | B3')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE', help='configs/base.yaml 의 값을 덮어쓴다 (여러 번 가능)')
    ap.add_argument('--config', default=C.BASE_YAML, help='기본 설정 파일')
    a = ap.parse_args(argv)
    cfg = C.load_config(a.dataset, a.preset, a.set, base_yaml=a.config)
    log(f'실행 이름: {cfg.run_name}  | 단계: {a.stage} | 결과 폴더: {cfg.run_dir}')
    STAGE_FUNCS[a.stage](cfg)


if __name__ == '__main__':
    main()
