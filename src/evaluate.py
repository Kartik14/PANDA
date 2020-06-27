import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.utils as vutils
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from model import enetv2
from data_loader import PANDADataset

device = torch.device('cuda')

def test(model, loader, criterion, df_valid):

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
    cm = confusion_matrix(TARGETS, PREDS)
    qwk_k = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'karolinska'], \
                              df_valid[df_valid['data_provider'] == 'karolinska'].isup_grade.values,\
                              weights='quadratic')         
    qwk_r = cohen_kappa_score(PREDS[df_valid['data_provider'] == 'radboud'], \
                              df_valid[df_valid['data_provider'] == 'radboud'].isup_grade.values,\
                              weights='quadratic')
    cm_k = confusion_matrix(PREDS[df_valid['data_provider'] == 'karolinska'], \
                              df_valid[df_valid['data_provider'] == 'karolinska'].isup_grade.values)                              
    cm_r = confusion_matrix(PREDS[df_valid['data_provider'] == 'radboud'], \
                              df_valid[df_valid['data_provider'] == 'radboud'].isup_grade.values)                              
        
    return val_loss, (qwk, qwk_k, qwk_r), (cm, cm_k, cm_r)

if __name__ == "__main__":
    
    data_dir = '../data/'
    df_biopsy = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    image_folder = os.path.join(data_dir, 'train_images')

    kernel_type = 'efficientnet-b0_144x128x128'
    enet_type = 'efficientnet-b0'
    num_folds = 5
    fold = 0
    tile_size = 128
    n_tiles = 144
    batch_size = 12
    num_workers = 4
    out_dim = 5

    skf = StratifiedKFold(num_folds, shuffle=True, random_state=42)
    df_biopsy['fold'] = -1
    for i, (train_idx, valid_idx) in enumerate(skf.split(df_biopsy, df_biopsy['isup_grade'])):
        df_biopsy.loc[valid_idx, 'fold'] = i

    model_path = sys.argv[1]
    model = enetv2(enet_type, out_dim)
    model.to(device)
    model = nn.DataParallel(model)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model saved at {model_path}")

    mean = [0.90949707, 0.8188697, 0.87795304]
    std = [0.36357649, 0.49984502, 0.40477625]
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    df_valid = df_biopsy.loc[df_biopsy['fold'] == fold]
    dataset_valid = PANDADataset(df_valid, image_folder, tile_size, n_tiles, \
        out_dim, transform=transform_val)
    valid_loader = DataLoader(dataset_valid,
                            batch_size=batch_size,
                            sampler=SequentialSampler(dataset_valid),
                            num_workers=num_workers)

    criterion = nn.BCEWithLogitsLoss()
    print("Number of validation samples : {}".format(len(dataset_valid)))
    
    val_loss, (qwk, qwk_k, qwk_r), (cm, cm_k, cm_r) = test(model, valid_loader, criterion, df_valid)
    print('qwk', qwk, 'qwk_k', qwk_k, 'qwk_r', qwk_r)

    print(cm)
    print(cm_k)
    print(cm_r)

    sns.set(font_scale=1.4) # for label size
    sns.heatmap(cm, annot=True, annot_kws={"size": 8}, fmt='d') # font size
    plt.title(f'N={n_tiles}, d={tile_size}, qwk={qwk:.4f}')
    plt.savefig(f'cm-{kernel_type}.png')


