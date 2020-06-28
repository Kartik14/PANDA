import os
import time
import skimage
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False
    
from model import enetv2
from data_loader import PANDADataset

device = torch.device('cuda')

class RotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = np.random.choice(self.angles)
        return TF.rotate(x, angle)

def train_epoch(model, loader, optimizer, criterion, use_amp=False):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    return train_loss


def val_epoch(model, loader, criterion, df_valid):

    model.eval()
    val_loss = []
    PREDS = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)

            loss = criterion(logits, target)

            pred = logits.sigmoid().sum(1).detach().round()            
            PREDS.append(pred)
            TARGETS.append(target.sum(1))

            val_loss.append(loss.detach().cpu().numpy())
        val_loss = np.mean(val_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    acc = (PREDS == TARGETS).mean() * 100.
    
    qwk = cohen_kappa_score(PREDS, TARGETS, weights='quadratic')
    qwk_k = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'karolinska'], \
                              df_valid[df_valid['data_provider'] == 'karolinska'].isup_grade.values,\
                              weights='quadratic')
    qwk_r = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'radboud'], \
                              df_valid[df_valid['data_provider'] == 'radboud'].isup_grade.values,\
                              weights='quadratic')
    print('qwk', qwk, 'qwk_k', qwk_k, 'qwk_r', qwk_r)
        
    return val_loss, acc, (qwk, qwk_k, qwk_r)

def main():

    data_dir = '../data/'
    df_biopsy = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    image_folder = os.path.join(data_dir, 'train_images')

    kernel_type = 'efficientnet-b3_36x256x256'
    enet_type = 'efficientnet-b3'
    num_folds = 5
    fold = 0
    tile_size = 256
    n_tiles = 32
    batch_size = 9
    num_workers = 24
    out_dim = 5
    init_lr = 3e-4
    warmup_factor = 10
    warmup_epo = 1
    n_epochs = 30
    use_amp = True

    writer = SummaryWriter(f'tensorboard_logs/{kernel_type}/fold-{fold}')

    if use_amp and not APEX_AVAILABLE:
        print("Error: could not import APEX module")
        exit()

    skf = StratifiedKFold(num_folds, shuffle=True, random_state=42)
    df_biopsy['fold'] = -1
    for i, (train_idx, valid_idx) in enumerate(skf.split(df_biopsy, df_biopsy['isup_grade'])):
        df_biopsy.loc[valid_idx, 'fold'] = i

    mean = [0.90949707, 0.8188697, 0.87795304]
    std = [0.36357649, 0.49984502, 0.40477625]
    transform_train = transforms.Compose([
        transforms.RandomChoice([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            RotationTransform([90, -90])
        ]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    df_train = df_biopsy.loc[df_biopsy['fold'] != fold]
    df_valid = df_biopsy.loc[df_biopsy['fold'] == fold]

    dataset_train = PANDADataset(df_train, image_folder, tile_size, n_tiles, \
        out_dim, transform=transform_train)
    dataset_valid = PANDADataset(df_valid, image_folder, tile_size, n_tiles, \
        out_dim, transform=transform_val)

    train_loader = DataLoader(dataset_train, 
                            batch_size=batch_size, 
                            sampler=RandomSampler(dataset_train),
                            num_workers=num_workers,)
    valid_loader = DataLoader(dataset_valid,
                            batch_size=batch_size,
                            sampler=SequentialSampler(dataset_valid),
                            num_workers=num_workers)

    model = enetv2(enet_type, out_dim=out_dim)    
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=init_lr/warmup_factor)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs-warmup_epo)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=warmup_factor, \
                                    total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

    criterion = nn.BCEWithLogitsLoss()

    if use_amp:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O1", 
            keep_batchnorm_fp32=None, loss_scale="dynamic"
        )
    model = nn.DataParallel(model)

    print("Number of train samples : {}".format(len(dataset_train)))
    print("Number of validation samples : {}".format(len(dataset_valid)))

    best_model = f'{kernel_type}_fold-{fold}_best.pth'
    save_path = f'../trained_models/{kernel_type}/fold-{fold}/'
    os.makedirs(save_path, exist_ok=True)

    qwk_max = 0.
    for epoch in range(1, n_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)
        scheduler.step(epoch-1)

        train_loss = train_epoch(model, train_loader, optimizer, criterion, use_amp=use_amp)
        val_loss, acc, (qwk, qwk_k, qwk_r) = val_epoch(model, valid_loader, criterion, df_valid)

        writer.add_scalars(f'loss', {'train' : np.mean(train_loss), 'val' : val_loss}, epoch)
        writer.add_scalars(f'qwk', {'total' :qwk, 'Karolinska' : qwk_k, 'Radboud': qwk_r}, epoch)
        content = "{}, Epoch {}, lr: {:.7f}, train loss: {:.5f}," \
                " val loss: {:.5f}, acc: {:.5f}, qwk: {:.5f}".format(
                    time.ctime(), epoch, optimizer.param_groups[0]["lr"], 
                    np.mean(train_loss), np.mean(val_loss), acc, qwk
                )
        print(content)
        
        with open('train_logs/log_{}_fold-{}.txt'.format(kernel_type, fold), 'a') as appender:
            appender.write(content + '\n')

        if qwk > qwk_max:
            print('score2 ({:.6f} --> {:.6f}).  Saving current best model ...'.format(qwk_max, qwk))
            torch.save(model.state_dict(), os.path.join(save_path, best_model))
            qwk_max = qwk

        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'qwk_max' : qwk_max  
        }, os.path.join(save_path, f'{kernel_type}_fold-{fold}_{epoch}.pth'))

if __name__ == '__main__':
    main()
