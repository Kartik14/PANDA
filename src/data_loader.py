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

        final_image = self.get_tiled_image(img_id)
        
        if self.transform is not None:
            final_image = self.transform(final_image)

        label = np.zeros(self.num_classes).astype(np.float32)
        label[:row.isup_grade] = 1.
        return final_image, torch.tensor(label)


        
        
        

