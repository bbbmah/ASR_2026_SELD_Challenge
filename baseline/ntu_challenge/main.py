# [변경 이력 시작]
#   2026-10-13  최초 생성: NTU 원본 main.py를 우리 데이터셋(main20, ctrl5) 학습용으로 다시 씀(zip 풀기, 특징, 학습, 매 에폭 val 채점으로 최고 모델 저장). wandb, torchinfo 제거
#   2026-10-13  --stage predict 추가: 저장된 최고 모델로 test 예측과 제출용 zip 만들기(채점 없음), 특징 옵션은 모델 파일에서 가져옴
# [변경 이력 끝]
"""
main.py

NTU 저장소(itsjunwei/NTU_SNTL_Task3)의 main.py 를 우리 챌린지 데이터셋(main20, ctrl5)으로 학습하도록 다시 쓴 진입점이다.
하는 일: --stage train (기본): (1) zip 풀기(선택) (2) 특징 뽑기 (3) 학습 (4) 매 에폭 val 채점으로 최고 모델 저장.
         --stage predict: 저장된 최고 모델로 test 를 예측하고 제출용 zip 을 만든다(채점은 하지 않는다. test 정답은 비공개).

예) python main.py --dataset main20 --data_root /content/data --out_root /content/drive/MyDrive/.../seld_runs \
                   --zip_dir /content/drive/MyDrive/.../dataset/main20
    python main.py --dataset ctrl5 --data_root ... --out_root ... --zip_dir .../internal --ms --iv --gamma --itfm

원본 학습 방식은 그대로다: AdamW(가중치 감쇠 1e-4, bias/BN/LN 은 제외), OneCycleLR, 기울기 누적, 100에폭.
원본에서 바꾼 점은 README_ntu.md 에 정리했다.
"""

import os
import sys
import csv
import json
import time
import shutil
import zipfile
import argparse
import platform
import tempfile
import numpy as np
import torch
from torch.utils.data import DataLoader

from parameters import params
from model import SELDModel
from loss import SELDLossADPIT, SELDLossSingleACCDOA
from data_generator import DataGenerator
from extract_features import SELDFeatureExtractor
from score import score_dirs
import training_utils as tu
import utils

LABEL_LEN = {'main20': 200, 'ctrl5': 50}        # 100ms 라벨 프레임 수
FRAMES_PER_BATCH = 12800                         # accum_batch=auto: 12800 / 라벨 프레임 수 (main20 64, ctrl5 256)


def log(*a):
    print(time.strftime('[%H:%M:%S]'), *a, flush=True)


# ---------------------------------------------------------------------- zip 풀기
def zip_path(zip_dir, dataset, split):
    name = {'train': 'train.zip', 'val': 'val.zip', 'test': 'test_inputs.zip'}[split] if dataset == 'main20' else f'internal_{dataset}_{split}.zip'
    return os.path.join(zip_dir, name)


def ensure_split(zip_dir, data_root, dataset, split):
    """zip 을 data_root/{dataset}/ 에 푼다. 이미 풀려 있으면 건너뛴다. zip 안에는 {split}/audio, {split}/labels, ... 가 들어 있다."""
    marker = os.path.join(data_root, '.done', f'{dataset}_{split}')
    if os.path.exists(marker):
        log(f'  [건너뜀] {dataset}/{split} 은 이미 풀려 있음')
        return
    zp = zip_path(zip_dir, dataset, split)
    if not os.path.exists(zp):
        raise FileNotFoundError(f'zip 을 찾을 수 없습니다: {zp}')
    log(f'  풀기: {zp}')
    with zipfile.ZipFile(zp) as z:
        z.extractall(os.path.join(data_root, dataset))
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, 'w').write(str(time.time()))


# ---------------------------------------------------------------------- 학습 도우미
def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def feat_tag():
    return (f"mel{params['nb_mels']}" if params['nb_mels'] else 'linspec') + ('_gamma' if params['gamma'] else '') + \
           ('_ipd' if params['ipd'] else '') + ('_iv' if params['iv'] else '') + ('_slite' if params['slite'] else '') + \
           ('_ms' if params['ms'] else '') + ('_dnorm' if params['dnorm'] else '')


def make_transform():
    n_aug_channels = 4 if params['ms'] else 2
    if params['cutout'] or params['itfm']:
        return tu.CompositeCutoutTorch(always_apply=False, p=0.5, aug_channels=n_aug_channels, use_itfm=params['itfm'])
    if params['freqshift']:
        return tu.RandomShiftUpDownTorch(always_apply=False, p=0.5, aug_channels=n_aug_channels)
    if params['filtaug']:
        return tu.FilterAugmentNormalized(always_apply=False, p=0.5, aug_channels=n_aug_channels)
    if params['compfreq']:
        return tu.CompositeFrequencyTorch(always_apply=False, p=0.5, aug_channels=n_aug_channels)
    if params['alltrans']:
        return tu.CompositeEverythingTorch(always_apply=False, p=0.5, aug_channels=n_aug_channels, use_itfm=params['itfm'])
    return None


def build_optimizer(model):
    if params['weight_decay'] != 0:
        no_decay = ['bias', 'bn.weight', 'bn.bias', 'ln.weight', 'ln.bias']
        decay, no_dec = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (no_dec if any(nd in name.lower() for nd in no_decay) else decay).append(p)
        groups = [{'params': decay, 'weight_decay': params['weight_decay']}, {'params': no_dec, 'weight_decay': 0.0}]
        return torch.optim.AdamW(groups, lr=params['learning_rate'])
    return torch.optim.Adam(params=model.parameters(), lr=params['learning_rate'])


def train_epoch(model, loader, optimizer, loss_fn, scheduler, device, accum_steps):
    model.train()
    losses, n = [], len(loader)
    optimizer.zero_grad()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y) / accum_steps
        loss.backward()
        losses.append(loss.item() * accum_steps)
        if (i + 1) % accum_steps == 0 or (i + 1) == n:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return float(np.mean(losses))


def val_epoch(model, ds, loss_fn, device, pred_dir):
    """val 예측 csv 를 pred_dir 에 쓰고 (val 손실, 채점 결과)를 돌려준다."""
    model.eval()
    if os.path.isdir(pred_dir):
        shutil.rmtree(pred_dir)
    os.makedirs(pred_dir)
    loader = DataLoader(ds, batch_size=params['val_batch_size'], shuffle=False, num_workers=0)
    tot, start = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tot += loss_fn(logits, y).item()
            utils.write_logits_to_dcase_format(logits, params, pred_dir, ds.names[start:start + logits.size(0)])
            start += logits.size(0)
    ref = os.path.join(params['data_root'], params['dataset'], 'val', 'labels')
    res = score_dirs(pred_dir, ref, params['label_sequence_length'], 'audio', None,
                     exclude_absent_classes=bool(params['exclude_absent_classes']))
    return tot / len(loader), res


def metric_value(res):
    v = res['ext' if params['select_metric'] == 'F_ext' else 'basic']['F']
    return float('-inf') if v is None else v


# ---------------------------------------------------------------------- predict (test 예측)
def predict(args, run_name, run_dir):
    """저장된 최고 모델(best_model.pth)로 test 를 예측해 pred_test/ 와 제출용 zip 을 만든다. 채점은 하지 않는다(test 정답은 비공개).
    특징 옵션(ms, gamma, iv 등)은 학습 때 값을 체크포인트에서 가져온다."""
    ck_path = os.path.join(run_dir, 'best_model.pth')
    if not os.path.exists(ck_path):
        raise FileNotFoundError(f'학습된 모델이 없습니다: {ck_path}  (같은 dataset, seed, exp 로 먼저 학습하세요)')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    for k in ('ms', 'gamma', 'iv', 'ipd', 'slite', 'nb_mels', 'max_freq', 'dnorm', 'multiACCDOA'):
        params[k] = ck['params'][k]
    ds_name = params['dataset']
    params['feat_dir'] = os.path.join(params['out_root'], 'features', ds_name, 'ntu_' + feat_tag())
    log(f'== test 예측: {run_name}  (최고 에폭 {ck["epoch"] + 1}, device={device})')
    if args.zip_dir:
        ensure_split(args.zip_dir, params['data_root'], ds_name, 'test')
    SELDFeatureExtractor(params).extract(('test',), labels=False)
    ds = DataGenerator(params, 'test', with_labels=False)
    model = SELDModel(params=params, in_feat_shape=(int(params['batch_size']),) + tuple(ds[0].shape)).to(device)
    model.load_state_dict(ck['seld_model'])
    model.eval()
    out = os.path.join(run_dir, 'pred_test')
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    loader = DataLoader(ds, batch_size=params['val_batch_size'], shuffle=False, num_workers=0)
    start = 0
    with torch.no_grad():
        for x in loader:
            logits = model(x.to(device))
            utils.write_logits_to_dcase_format(logits, params, out, ds.names[start:start + logits.size(0)])
            start += logits.size(0)
    zpath = os.path.join(run_dir, f'submission_{run_name}.zip')
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out)):
            z.write(os.path.join(out, f), f)
    log(f'test 예측 {len(ds)}개 -> {out}\n제출용 zip: {zpath}')


# ---------------------------------------------------------------------- main
def main(args):
    for k, v in vars(args).items():
        if v is not None:
            params[k] = v
    ds_name = params['dataset']
    params['label_sequence_length'] = LABEL_LEN[ds_name]
    if platform.system() == 'Windows':
        params['nb_workers'] = 0

    run_name = f"ntu_{ds_name}_s{params['seed']}" + (f"_{args.exp}" if args.exp else '')
    run_dir = os.path.join(params['out_root'], 'runs', run_name)
    params['feat_dir'] = os.path.join(params['out_root'], 'features', ds_name, 'ntu_' + feat_tag())
    os.makedirs(run_dir, exist_ok=True)
    if args.stage == 'predict':
        return predict(args, run_name, run_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(params['seed'])

    # accum_batch: 가중치를 한 번 갱신하는 클립 수
    accum = FRAMES_PER_BATCH // params['label_sequence_length'] if params['accum_batch'] == 'auto' else int(params['accum_batch'])
    bs = int(params['batch_size'])
    if accum % bs != 0:
        raise ValueError(f'accum_batch({accum}) 는 batch_size({bs}) 의 배수여야 합니다')
    accum_steps = accum // bs
    log(f'== {run_name}  (dataset={ds_name}, 라벨 프레임 {params["label_sequence_length"]}, 갱신 단위 {accum}클립 = {bs} x {accum_steps}, device={device})')

    # 1) zip 풀기, 2) 특징
    if args.zip_dir:
        for sp in ('train', 'val'):
            ensure_split(args.zip_dir, params['data_root'], ds_name, sp)
    log(f'특징 폴더: {params["feat_dir"]}')
    SELDFeatureExtractor(params).extract(('train', 'val'))

    # 3) 데이터
    train_ds = DataGenerator(params, 'train', transform=make_transform())
    val_ds = DataGenerator(params, 'val')
    loader = DataLoader(train_ds, batch_size=bs, shuffle=params['shuffle'], num_workers=params['nb_workers'], drop_last=False, pin_memory=True)
    x0, y0 = train_ds[0]
    in_shape = (bs,) + tuple(x0.shape)
    log(f'train {len(train_ds)}개, val {len(val_ds)}개, 입력 {tuple(x0.shape)}, 라벨 {tuple(y0.shape)}, 메모리 약 {train_ds.ram_gb() + val_ds.ram_gb():.1f}GB')

    # 4) 모델, 손실, 최적화
    model = SELDModel(params=params, in_feat_shape=in_shape).to(device)
    log(f'모델 파라미터 {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M')
    loss_fn = (SELDLossADPIT(params=params) if params['multiACCDOA'] else SELDLossSingleACCDOA(params=params)).to(device)
    optimizer = build_optimizer(model)
    steps_per_epoch = int(np.ceil(len(loader) / accum_steps))
    total_steps = params['nb_epochs'] * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=params['learning_rate'], total_steps=total_steps)
    log(f'에폭당 갱신 {steps_per_epoch}번, {params["nb_epochs"]}에폭 = {total_steps}번 (OneCycleLR)')
    json.dump({k: (str(v) if isinstance(v, float) and v == float('inf') else v) for k, v in params.items()},
              open(os.path.join(run_dir, 'config_used.json'), 'w', encoding='utf8'), ensure_ascii=False, indent=1)

    pred_dir = os.path.join(args.local_tmp or tempfile.gettempdir(), run_name, 'pred_val')
    log_path = os.path.join(run_dir, 'train_log.csv')
    cols = ['epoch', 'train_loss', 'val_loss', 'F_ext', 'F_basic', 'DOA_ext', 'DOA_basic', 'rel_dist_ext', 'seconds']
    if os.path.exists(log_path):
        os.remove(log_path)
    best_val, best_epoch, best_res = float('-inf'), -1, None
    for epoch in range(params['nb_epochs']):
        t0 = time.time()
        train_loss = train_epoch(model, loader, optimizer, loss_fn, scheduler, device, accum_steps)
        row = dict(epoch=epoch + 1, train_loss=train_loss)
        if epoch % params['val_freq'] == 0 or epoch / params['nb_epochs'] > 0.8:
            val_loss, res = val_epoch(model, val_ds, loss_fn, device, pred_dir)
            e, b = res['ext'], res['basic']
            row.update(val_loss=val_loss, F_ext=e['F'], F_basic=b['F'], DOA_ext=e['DOA_err'], DOA_basic=b['DOA_err'], rel_dist_ext=e['rel_dist_err'])
            cur = metric_value(res)
            if cur >= best_val:
                best_val, best_epoch, best_res = cur, epoch, res
                torch.save({'seld_model': model.state_dict(), 'epoch': epoch, 'params': params, 'best_value': best_val,
                            'select_metric': params['select_metric'], 'val_scores': res}, os.path.join(run_dir, 'best_model.pth.tmp'))
                os.replace(os.path.join(run_dir, 'best_model.pth.tmp'), os.path.join(run_dir, 'best_model.pth'))
            pct = lambda v: 'NA' if v is None else f'{100 * v:.2f}'
            log(f"에폭 {epoch + 1}/{params['nb_epochs']} | 학습손실 {train_loss:.4f} | val손실 {val_loss:.4f} | F_ext {pct(e['F'])} F_basic {pct(b['F'])} | "
                f"최고 {pct(best_val)} (에폭 {best_epoch + 1}) | {time.time() - t0:.0f}초")
        else:
            log(f"에폭 {epoch + 1}/{params['nb_epochs']} | 학습손실 {train_loss:.4f} | {time.time() - t0:.0f}초")
        row['seconds'] = time.time() - t0
        new = not os.path.exists(log_path)
        with open(log_path, 'a', newline='', encoding='utf8') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerow(row)

    json.dump(dict(run=run_name, best_epoch=best_epoch + 1, best_value=best_val, select_metric=params['select_metric'],
                   epochs_done=params['nb_epochs'], steps_per_epoch=steps_per_epoch, total_updates=total_steps,
                   train_clips=len(train_ds), val_scores=best_res),
              open(os.path.join(run_dir, 'train_summary.json'), 'w', encoding='utf8'), ensure_ascii=False, indent=1)
    log(f"== 학습 끝. 최고 {params['select_metric']}={best_val:.4f} (에폭 {best_epoch + 1}). 모델: {os.path.join(run_dir, 'best_model.pth')}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='NTU 시스템을 챌린지 데이터셋(main20, ctrl5)으로 학습', formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=list(LABEL_LEN), default='main20')
    ap.add_argument('--data_root', required=True, help='{data_root}/{dataset}/{train,val}/{audio,labels}/ 가 있는(또는 zip 을 풀) 곳')
    ap.add_argument('--out_root', required=True, help='결과(runs/)와 특징 캐시(features/)를 저장할 곳')
    ap.add_argument('--stage', choices=['train', 'predict'], default='train', help='train = 학습(기본), predict = 저장된 최고 모델로 test 예측과 제출용 zip 만들기')
    ap.add_argument('--zip_dir', default=None, help='train/val zip 이 있는 곳(main20: train.zip, val.zip / ctrl5: internal_ctrl5_train.zip ...). 주면 data_root 에 풀어 준다')
    ap.add_argument('--local_tmp', default=None, help='에폭마다 쓰는 val 예측 csv 위치(로컬 디스크). 기본은 시스템 임시 폴더')
    ap.add_argument('--exp', default='', help='실행 이름 꼬리표')
    ap.add_argument('--seed', type=int)
    ap.add_argument('--nb_epochs', type=int)
    ap.add_argument('--batch_size', type=int, help='한 번에 올리는 클립 수')
    ap.add_argument('--accum_batch', help='갱신 단위 클립 수. auto = 12800 / 라벨 프레임 수')
    ap.add_argument('--learning_rate', type=float)
    ap.add_argument('--weight_decay', type=float)
    ap.add_argument('--dropout', type=float)
    ap.add_argument('--val_freq', type=int)
    ap.add_argument('--select_metric', choices=['F_ext', 'F_basic'])
    # 특징과 증강 (NTU 원본 옵션. 기본은 모두 꺼짐)
    for flag, h in [('ms', 'Mid-Side 스펙트로그램'), ('gamma', '채널 간 coherence'), ('ipd', '채널 간 위상차'), ('iv', 'Mid-Side intensity vector'),
                    ('slite', 'SALSA-Lite'), ('cutout', '시간-주파수 마스킹'), ('itfm', 'Inter-channel level-aware 마스킹'),
                    ('freqshift', '주파수 이동'), ('filtaug', 'FilterAugment'), ('compfreq', '주파수 이동+FilterAugment'), ('alltrans', '위 증강 모두')]:
        ap.add_argument('--' + flag, action='store_true', default=None, help=h)
    a = ap.parse_args()
    try:
        sys.exit(main(a))
    except (ValueError, IOError) as e:
        sys.exit(e)
