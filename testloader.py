import os
import sys
import json

import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from PIL import Image

np.random.seed(0)

from torchvision import transforms  # noqa

class WSIimageDatasetpredict(Dataset):

    def __init__(self,slide,level,normalize=True):
        #self._data_path = data_path
        self._slide = slide
        self.level = level
        self._normalize = normalize
        with open('dataconfig/slideindex2.json') as f:
            self.slideindex = json.load(f)
        self._pre_process()

    def _pre_process(self):
        
        #classes = ['tumor','normal'] #仅两类
        #class_to_idx = {'tumor':1,'normal':0}
        # make dataset
        self.imgs = []

        patchdir = os.path.join('patchdata',self._slide)
        imgpath = np.load(os.path.join(patchdir,f'lv{self.level}_uniform_imglist.npy'))
        #print(imgpath)

        index = np.array(self.slideindex[self._slide])
        #index = np.random.choice(np.arange(len(imgpath)),size=100,replace=False)
        imgpath = imgpath[index]

        self.imgs.extend(imgpath.tolist())

        self.imgs = np.array(self.imgs)


        self._num_images = len(self.imgs)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path = self.imgs[idx]
        img = Image.open(path)

        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        return img

class WSIimageDatasetpredict2(Dataset):

    def __init__(self,slide,level,normalize=True):
        #self._data_path = data_path
        self._slide = slide
        self.level = level
        self._normalize = normalize
        self._pre_process()

    def _pre_process(self):
        
        #classes = ['tumor','normal'] #仅两类
        #class_to_idx = {'tumor':1,'normal':0}
        # make dataset
        self.imgs = []
        datalist = os.listdir('wsidata_aug_uniform')
        for img in datalist:
            patch = img.split('.')[0]
            slide = patch.split('_')[0]
            if slide == self._slide:
                imgpath = os.path.join('wsidata_aug_uniform',img)
                self.imgs.append(imgpath)

        #self.imgs.extend(imgpath.tolist())

        self.imgs = np.array(self.imgs)


        self._num_images = len(self.imgs)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path = self.imgs[idx]
        img = Image.open(path)

        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        return img