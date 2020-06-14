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
        self.mode = mode

    def __len__(self):
        return self.df.shape[0]
    
    def get_tiled_image(self, img_id):
        tiles = []
        for i in range(self.n_tiles):
            img_path = os.path.join(self.image_folder, \
                                    '{}_{}.jpeg'.format(img_id, i))
            tiles.append(io.imread(img_path))

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

        if self.mode == 'tile':
            final_image = self.get_tiled_image(img_id)
        else:
            final_image = io.imread(os.path.join(self.image_folder, img_id + '_2.jpeg'))
            final_image = Image.fromarray(final_image)
            final_image = f.resize(final_image, 256)
        
        if self.transform is not None:
            final_image = self.transform(final_image)

        label = np.zeros(self.num_classes).astype(np.float32)
        label[:row.isup_grade] = 1.
        return final_image, torch.tensor(label)

def collate_fn(batch):

    mean = [0.9580, 0.9179, 0.9430]
    std = [0.3196, 0.4319, 0.3588]
    min_h = min(batch, key=lambda x:x[0].size[0])[0].size[0]
    min_w = min(batch, key=lambda x:x[0].size[1])[0].size[1]

    targets = []
    images = []
    for sample in batch:
        image, target = sample
        image = f.center_crop(image, (min_h, min_w))
        image = f.normalize(f.to_tensor(image), mean, std)
        images.append(image)
        targets.append(target)

    return torch.stack(images), torch.stack(targets)

    

        
        
        

