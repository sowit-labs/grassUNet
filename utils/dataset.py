import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class PAIDataset(Dataset):
    """A dataset class suitable for the segmentation task with scaling and augmentation transformations"""
    def __init__(self, imgs_dir, masks_dir, scale=1, transform=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.transform = transform
        
        self.image_names = []
        for local_root, dirs, files in os.walk(imgs_dir):
            path = local_root.split(os.sep)
            for file in files:
                self.image_names.append(os.path.join(imgs_dir, os.path.relpath(local_root,imgs_dir), file))
        list.sort(self.image_names)
                
        self.mask_names = []
        for local_root, dirs, files in os.walk(masks_dir):
            path = local_root.split(os.sep)
            for file in files:
                self.mask_names.append(os.path.join(masks_dir, os.path.relpath(local_root,masks_dir), file))
        list.sort(self.mask_names)

    def __len__(self):
        return len(self.image_names)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img, dtype=np.dtype('uint8'))

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        img_file = self.image_names[i]
        mask_file = self.mask_names[i]

        img = Image.open(img_file)
        mask = Image.open(mask_file).convert('1')
        
        assert img.size == mask.size, \
            f'Image and mask {i} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        
        sample = {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample