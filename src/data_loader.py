import os
from skimage import io
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as f
from torch.utils.data import Dataset

def tile(img, sz, pixel_th=0.1, min_tiles=5):
    result = []
    shape = img.shape
    pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
    img = np.pad(img, [[pad0 // 2,pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], \
                      [0,0]], constant_values=255)
    rows = img.shape[0] / sz
    cols = img.shape[1] / sz                   
    img = img.reshape(img.shape[0] // sz, sz, img.shape[1] // sz, sz, 3)
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, sz, sz, 3)
    max_sum = 255*sz*sz*3
    img_sums = img.reshape(img.shape[0],-1).sum(-1)
    topk = (img_sums < (1-pixel_th)*max_sum).sum()
    topk = max(topk, min(min_tiles, img.shape[0]))
    idxs = np.argsort(img_sums)[:topk]
    img = img[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'idx':idxs[i]})
    return result, (rows, cols)

class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 tile_size,
                 image_folder,
                 num_classes,
                 mode,
                 transform=None,
                 tile_drop_prob=0.2,                 
                ):

        self.df = df.reset_index(drop=True)
        self.tile_size = tile_size
        self.image_folder = image_folder
        self.transform = transform
        self.num_classes = num_classes
        self.mode = mode
        self.tile_drop_prob = tile_drop_prob

    def __len__(self):
        return self.df.shape[0]

    def drop_tiles(self, tiles):
        drop_val = np.random.binomial(1, self.tile_drop_prob, size=len(tiles))
        while drop_val.sum() == len(tiles):
            drop_val = np.random.binomial(1, self.tile_drop_prob, size=len(tiles))
        return [t for t, d in zip(tiles, drop_val) if d == 0]
    
    def get_image_tiles(self, img_id):
        tiles = []
        img_path = os.path.join(self.image_folder, '{}_2.jpeg'.format(img_id))
        img = io.imread(img_path)
        tiles, (rows, cols) = tile(img, sz=self.tile_size)
        if self.mode == 'train':
            tiles = self.drop_tiles(tiles)
        idxs = [res['idx'] for res in tiles]
        tiles = [Image.fromarray(res['img']) for res in tiles]

        return tiles, idxs, (rows, cols)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id

        tiles, idxs, (rows, cols) = self.get_image_tiles(img_id)   
        norm_tiles = []
        if self.transform is not None:
            for tile in tiles:
                norm_tiles.append(self.transform(tile))
        assert len(norm_tiles) != 0, print("fml")
        norm_tiles = torch.stack(norm_tiles)
        label = [0.0] if row.isup_grade == 0 else [1.0]
        return norm_tiles, torch.tensor(label), idxs, (img_id, (rows, cols))

def collate_fn(batch):

    images = []
    num_tiles = []
    tile_indices = []
    targets = []
    img_metadata = []

    for sample in batch:
        images.append(sample[0])
        num_tiles.append(len(sample[0]))
        targets.append(sample[1])
        tile_indices.append(sample[2])
        img_metadata.append(sample[3])

    img_dim = images[0].shape # N x 3 x 256 x 256
    images = torch.cat(images).reshape(-1, img_dim[-3], img_dim[-2], img_dim[-1])
    targets = torch.stack(targets)
    
    return images, num_tiles, targets, tile_indices, img_metadata