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



class TestimageDataset(Dataset):

    def __init__(self,slide_list,img_size,level,is_training=True,
                 crop_size=512, normalize=True):
        #self._data_path = data_path
        self._slide_list = slide_list
        self._img_size = img_size
        self.level = level
        self._crop_size = crop_size
        self._normalize = normalize
        self.is_training = is_training
        # 作色调、饱和度等调整
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        with open('cellfeatures.json','r') as f:
            self.slidecf = json.load(f)
        self._pre_process()

    def _pre_process(self):
        
        #classes = ['tumor','normal'] #仅两类
        #class_to_idx = {'tumor':1,'normal':0}
        # make dataset
        self.imgs = []
        self.labels_nf = []
        self.labels_shape = []
        # 中间还要加一个切片名的目录，加一个全部切片名的成员，前面有保存的
        if self.is_training:
            #print(self._slide_list)
            for slide in self._slide_list:
                patchdir = os.path.join('patchdata',slide)
                imgpath = np.load(os.path.join(patchdir,f'lv{self.level}_trainlist.npy'))
                label_nf = np.zeros((len(imgpath),4))
                label_shape = np.zeros((len(imgpath),5))
                # to onehot
                label_nf[:,self.slidecf[slide][1]]=1
                label_shape[:,self.slidecf[slide][2]]=1

                self.imgs.extend(imgpath.tolist())
                self.labels_nf.extend(label_nf.tolist())
                self.labels_shape.extend(label_shape.tolist())

                #random.shuffle(self.imgs)
                #random.shuffle(self.labels_nf)
                #random.shuffle(self.labels_shape)

        else:
            for slide in self._slide_list:
                patchdir = os.path.join('patchdata',slide)
                imgpath = np.load(os.path.join(patchdir,f'lv{self.level}_vallist.npy'))
                label_nf = np.zeros((len(imgpath),4))
                label_shape = np.zeros((len(imgpath),5))
                # to onehot
                label_nf[:,self.slidecf[slide][1]]=1
                label_shape[:,self.slidecf[slide][2]]=1

                self.imgs.extend(imgpath.tolist())
                self.labels_nf.extend(label_nf.tolist())
                self.labels_shape.extend(label_shape.tolist())

        # shuffle data
        index = np.arange(len(self.imgs))
        random.shuffle(index)
        self.imgs = np.array(self.imgs)[index]
        self.labels_nf = np.array(self.labels_nf)[index]
        self.labels_shape = np.array(self.labels_shape)[index]

        self._num_images = len(self.imgs)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path = self.imgs[idx]
        label_nf = self.labels_nf[idx]
        label_shape = self.labels_shape[idx]
        img = Image.open(path)

        ### data augmentation
        # color jitter
        img = self._color_jitter(img)

        # use left_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # use rotate
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)

        # PIL image: H W C
        # torch image: C H W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        return img, label_nf, label_shape

#class test_WSIimageDataset(Dataset):



class tumortestImageDataset(Dataset):

    def __init__(self,slide_list,img_size,
                 crop_size=224, normalize=True):
        #self._data_path = data_path
        self._slide_list = slide_list
        self._img_size = img_size
        self._crop_size = crop_size
        self._normalize = normalize
        # 作色调、饱和度等调整
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self._pre_process()

    def _pre_process(self):
        '''
        # find classes
        if sys.version_info >= (3, 5):
            # Faster and available in python 3.5 and above
            classes = [d.name for d in os.scandir(self._data_path) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._data_path) if os.path.isdir(os.path.join(self._data_path, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        '''
        classes = ['tumor','normal'] #仅两类
        class_to_idx = {'tumor':1,'normal':0}
        # make dataset
        self._items = []

        # 中间还要加一个切片名的目录，加一个全部切片名的成员，前面有保存的
        for slide in self._slide_list:
            for patchcls in classes:
                patches = os.listdir(os.path.join('data',slide,patchcls))
                for patch in patches:
                    imgpath = os.path.join('data',slide,patchcls,patch)
                    item = (imgpath,class_to_idx[patchcls])
                    self._items.append(item)
        '''
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(self._data_path, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.split('.')[-1] == 'png':
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        self._items.append(item)
        '''
        random.shuffle(self._items)

        self._num_images = len(self._items)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path, label = self._items[idx]
        # label = 1: tumor
        label = np.array(label, dtype=float)

        img = Image.open(path)

        ### data augmentation
        # color jitter
        img = self._color_jitter(img)

        # use left_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # use rotate
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)

        # PIL image: H W C
        # torch image: C H W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0

        return img, label