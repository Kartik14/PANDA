import os
from os.path import join, basename, dirname
import cv2
import skimage.io
from tqdm import tqdm
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import glob
import multiprocessing
from multiprocessing.pool import Pool
from PIL import Image
import pandas as pd

Image.MAX_IMAGE_PIXELS = None
sz = 64

def tile(img, pixel_th=0.1):
    result = []
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    img = np.pad(img, [[pad0 // 2,pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], \
                      [0,0]], constant_values=255)
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    # if len(img) < N:
    #     img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)

    max_sum = 255*sz*sz*3
    img_sums = img.reshape(img.shape[0],-1).sum(-1)
    topk = (img_sums < (1-pixel_th)*max_sum).sum()
    idxs = np.argsort(img_sums)[:topk]
    img = img[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'idx':idxs[i]})
    return result

def process_and_save_image(img_path):
    full_image = skimage.io.imread(img_path)
    tiles = tile(full_image)
    tiles = sorted(tiles, key=lambda x: x['idx'])
    for i, t in enumerate(tiles):
        img = t['img']
        save_path = join(OUT_TRAIN, basename(img_path[:-7]) + '_' + str(i) + '.jpeg')
        skimage.io.imsave(save_path, img, check_contrast=False)

if __name__ == "__main__":

    df = pd.read_csv('../data/train.csv').set_index('image_id')
    TRAIN = '../data/train_images/'
    OUT_TRAIN = '../data/train_images_tiles_64x64'
    os.makedirs(OUT_TRAIN, exist_ok=True)
    TRAIN_FILES = glob.glob(os.path.join(TRAIN, '*_2.jpeg'))
    
    with Pool(multiprocessing.cpu_count()) as pool:
        list(tqdm(pool.imap(process_and_save_image, TRAIN_FILES), total=len(TRAIN_FILES)))
        