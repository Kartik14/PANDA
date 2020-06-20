import os
from skimage import io
import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as f
from torch.utils.data import Dataset

class PANDADataset(Dataset):
    def __init__(self,
                 df,
                 image_folder,
                 n_tiles,
                 num_classes,
                 transform=None,
                 mode='tile'
                ):

        self.df = df.reset_index(drop=True)
        self.image_folder = image_folder
        self.n_tiles = n_tiles        
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return self.df.shape[0]
    
    def get_image_tiles(self, img_id):
        tiles = []
        for i in range(self.n_tiles):
            img_path = os.path.join(self.image_folder, \
                                    '{}_{}.jpeg'.format(img_id, i))
            tiles.append(Image.fromarray(io.imread(img_path)))

        return tiles

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = row.image_id

        tiles = self.get_image_tiles(img_id)
        
        norm_tiles = []
        if self.transform is not None:
            for tile in tiles:
                norm_tiles.append(self.transform(tile))
        norm_tiles = torch.stack(norm_tiles)

        label = np.zeros(self.num_classes).astype(np.float32)
        label[:row.isup_grade] = 1.
        return norm_tiles, torch.tensor(label)

def collate_fn(batch):

    images = []
    targets = []

    for sample in batch:
        images.append(sample[0])
        targets.append(sample[1])

    img_dim = images[0].shape # N x 3 x 256 x 256
    images = torch.stack(images).reshape(-1, img_dim[-3], img_dim[-2], img_dim[-1])
    targets = torch.stack(targets)
    
    return images, targets