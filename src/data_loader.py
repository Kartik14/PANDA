import os
from skimage import io
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as f
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None

def tile(img, sz, N):
    result = []
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    img = np.pad(img, [[pad0 // 2,pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], \
                      [0,0]], constant_values=255)
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    if len(img) < N:
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'idx':idxs[i]})
    return result

class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_folder,
                 tile_size,
                 n_tiles,
                 num_classes,
                 transform=None,
                ):

        self.df = df.reset_index(drop=True)
        self.image_folder = image_folder
        self.tile_size = tile_size
        self.n_tiles = n_tiles        
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return self.df.shape[0]
    
    def create_tiled_image(self, img_id):
        img_path = os.path.join(self.image_folder, f'{img_id}_1.jpeg')
        image = io.imread(img_path)
        tiles = tile(image, self.tile_size, self.n_tiles)
        # sorting tiles in row-major order
        tiles = sorted(tiles, key=lambda x: x['idx']) 
        tiles = [t['img'] for t in tiles]
        
        n_rows = int(np.sqrt(self.n_tiles))
        tiled_image = []
        for i in range(n_rows):
            tiled_image.append(np.concatenate(tiles[n_rows*i:n_rows*i \
                                                    + n_rows], axis=1))
        tiled_image = np.concatenate(tiled_image, axis=0)
        tiled_image = Image.fromarray(tiled_image)
        return tiled_image

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id
        tiled_image = self.create_tiled_image(img_id)
        
        if self.transform is not None:
            tiled_image = self.transform(tiled_image)

        label = np.zeros(self.num_classes).astype(np.float32)
        label[:row.isup_grade] = 1.
        return tiled_image, torch.tensor(label)