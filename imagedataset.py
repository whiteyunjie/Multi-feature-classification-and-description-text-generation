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

class ImageDataset(Dataset):

    def __init__(self,slide_list,img_size,level,samplerate,is_training=True,
                 crop_size=512, normalize=True):
        #self._data_path = data_path
        self._slide_list = slide_list
        self._img_size = img_size
        self.level = level
        self.samplerate  = samplerate
        self._crop_size = crop_size
        self._normalize = normalize
        self.is_training = is_training
        # 作色调、饱和度等调整
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        with open('cellfeaturesn.json','r') as f:
            self.slidecf = json.load(f)
        self._pre_process()

    def _pre_process(self):
        
        #classes = ['tumor','normal'] #仅两类
        #class_to_idx = {'tumor':1,'normal':0}
        # make dataset
        self.imgs = []
        self.labels_nf = []
        self.labels_ncr = []
        self.labels_shape = []
        # 中间还要加一个切片名的目录，加一个全部切片名的成员，前面有保存的
        if self.is_training:
            #print(self._slide_list)
            for slide in self._slide_list:
                patchdir = os.path.join('patchdata',slide)
                imgpath = np.load(os.path.join(patchdir,f'lv{self.level}_uniform_imglist.npy'))
                rate = self.samplerate[slide]
                samplesize = int(rate*200)
                # 限制每个切片的patch数防止过拟合
                index = np.random.choice(np.arange(len(imgpath)),size=samplesize,replace=False)
                imgpath = imgpath[index]

                label_nf = np.zeros((len(imgpath),2))
                label_ncr = np.zeros((len(imgpath),2))
                label_shape = np.zeros((len(imgpath),2))
                # to onehot
                label_nf[:,self.slidecf[slide][1]]=1
                label_ncr[:,self.slidecf[slide][2]]=1
                label_shape[:,self.slidecf[slide][3]]=1

                self.imgs.extend(imgpath.tolist())
                self.labels_nf.extend(label_nf.tolist())
                self.labels_ncr.extend(label_ncr.tolist())
                self.labels_shape.extend(label_shape.tolist())

                #random.shuffle(self.imgs)
                #random.shuffle(self.labels_nf)
                #random.shuffle(self.labels_shape)

        else:
            for slide in self._slide_list:
                patchdir = os.path.join('patchdata',slide)
                imgpath = np.load(os.path.join(patchdir,f'lv{self.level}_uniform_imglist.npy'))

                # 限制每个切片的patch数防止过拟合
                rate = self.samplerate[slide]
                samplesize = int(rate*150)
                # 限制每个切片的patch数防止过拟合
                index = np.random.choice(np.arange(len(imgpath)),size=samplesize,replace=False)
                imgpath = imgpath[index]

                label_nf = np.zeros((len(imgpath),2))
                label_ncr = np.zeros((len(imgpath),2))
                label_shape = np.zeros((len(imgpath),2))
                # to onehot
                label_nf[:,self.slidecf[slide][1]]=1
                label_ncr[:,self.slidecf[slide][2]]=1
                label_shape[:,self.slidecf[slide][3]]=1

                self.imgs.extend(imgpath.tolist())
                self.labels_nf.extend(label_nf.tolist())
                self.labels_ncr.extend(label_ncr.tolist())
                self.labels_shape.extend(label_shape.tolist())

        # shuffle data
        index = np.arange(len(self.imgs))
        random.shuffle(index)
        self.imgs = np.array(self.imgs)[index]
        self.labels_nf = np.array(self.labels_nf)[index]
        self.labels_ncr = np.array(self.labels_ncr)[index]
        self.labels_shape = np.array(self.labels_shape)[index]

        self._num_images = len(self.imgs)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path = self.imgs[idx]
        label_nf = self.labels_nf[idx]
        label_ncr = self.labels_ncr[idx]
        label_shape = self.labels_shape[idx]
        img = Image.open(path)

        ### data augmentation
        # color jitter
        #img = self._color_jitter(img)

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

        return img, label_nf, label_ncr, label_shape


class ShapeImageDataset(Dataset):

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
        with open('cellfeatures3.json','r') as f:
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
                imgpath = np.load(os.path.join(patchdir,f'lv{self.level}_uniform_trainlist.npy'))
                # 限制每个切片的patch数防止过拟合
                index = np.random.choice(np.arange(len(imgpath)),size=200,replace=False)
                imgpath = imgpath[index]

                label_nf = np.zeros((len(imgpath),2))
                label_shape = np.zeros((len(imgpath),2))
                # to onehot
                label_nf[:,self.slidecf[slide][1]]=1
                label_shape[:,self.slidecf[slide][3]]=1

                self.imgs.extend(imgpath.tolist())
                self.labels_nf.extend(label_nf.tolist())
                self.labels_shape.extend(label_shape.tolist())

                #random.shuffle(self.imgs)
                #random.shuffle(self.labels_nf)
                #random.shuffle(self.labels_shape)

        else:
            for slide in self._slide_list:
                patchdir = os.path.join('patchdata',slide)
                imgpath = np.load(os.path.join(patchdir,f'lv{self.level}_uniform_vallist.npy'))

                # 限制每个切片的patch数防止过拟合
                index = np.random.choice(np.arange(len(imgpath)),size=100,replace=False)
                imgpath = imgpath[index]

                label_nf = np.zeros((len(imgpath),2))
                label_shape = np.zeros((len(imgpath),2))
                # to onehot
                label_nf[:,self.slidecf[slide][1]]=1
                label_shape[:,self.slidecf[slide][3]]=1

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
        #img = self._color_jitter(img)

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


class WSIimageDataset(Dataset):

    def __init__(self,is_training=True,mission = 'envelop',
                 crop_size=512, normalize=True):

        self._crop_size = crop_size
        self._normalize = normalize
        self.is_training = is_training
        self.mission = mission
        # 作色调、饱和度等调整
        #self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        with open('feats.json','r') as f:
            self.slidecf = json.load(f)
        self._pre_process()

    def _pre_process(self):
        
        #classes = ['tumor','normal'] #仅两类
        #class_to_idx = {'tumor':1,'normal':0}
        # make dataset
        self.imgs = []
        self.labels_nec = []
        self.labels_env = []

        if self.is_training:
            self.imgs = np.load('wsiimg.npy')[:,0]
            for img in self.imgs:
                label_nec = np.zeros((2))
                label_env = np.zeros((2))
                # to onehot
                label_nec[self.slidecf[img][1]]=1
                label_env[self.slidecf[img][0]]=1

                self.labels_nec.append(label_nec)
                self.labels_env.append(label_env)

                index = np.arange(len(self.imgs))
                random.shuffle(index)
        else:
            self.imgs = np.load('wsival.npy')[:,0]
            for img in self.imgs:
                label_nec = np.zeros((2))
                label_env = np.zeros((2))
                # to onehot
                label_nec[self.slidecf[img][1]]=1
                label_env[self.slidecf[img][0]]=1

                self.labels_nec.append(label_nec)
                self.labels_env.append(label_env)

                index = np.arange(len(self.imgs))
        
        self.imgs = np.array(self.imgs)[index]
        self.labels_nec = np.array(self.labels_nec)[index]
        self.labels_env = np.array(self.labels_env)[index]
        #self.labels_shape = np.array(self.labels_shape)[index]
        self._num_images = len(self.imgs)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path = self.imgs[idx]
        label_nec = self.labels_nec[idx]
        label_env = self.labels_env[idx]
        # label = 1: tumor
        #label = np.array(label)
        ## convert to one-hot
        #labeloh = np.zeros((3,))
        #labeloh[label] = 1
        #label = labeloh.astype(float)

        img = Image.open(os.path.join('wsidata_aug_uniform',path))

        ### data augmentation
        # color jitter
        #img = self._color_jitter(img)

        # use left_right flip
        #if np.random.rand() > 0.5:
        #    img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # use rotate
        #num_rotate = np.random.randint(0, 4)
        #img = img.rotate(90 * num_rotate)

        # PIL image: H W C
        # torch image: C H W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        if self._normalize:
            img = (img - 128.0) / 128.0
        
        if self.mission == 'envelop':
            return img,label_env
        if self.mission == 'necrosis':
            return img, label_nec


class GradeImageDataset(Dataset):

    def __init__(self,slide_list,img_size,level,samplerate,is_training=True,
                 crop_size=224, normalize=True):
        #self._data_path = data_path
        self._slide_list = slide_list
        self._img_size = img_size
        self.level = level
        self.samplerate = samplerate
        self._crop_size = crop_size
        self._normalize = normalize
        self.is_training = is_training
        # 作色调、饱和度等调整
        self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        ### 用新标签更新slidelevel
        with open('cellfeaturesn.json','r') as f:
            self.slidefc = json.load(f)
        self._pre_process()

    def _pre_process(self):
        
        #classes = ['tumor','normal'] #仅两类
        #class_to_idx = {'tumor':1,'normal':0}
        # make dataset
        self.imgs = []
        self.labels = []
        # 中间还要加一个切片名的目录，加一个全部切片名的成员，前面有保存的
        if self.is_training:
            for slide in self._slide_list:
                patchdir = os.path.join('patchdata',slide)
                imgpath = np.load(os.path.join(patchdir,f'lv{self.level}_uniform_imglist.npy'))

                rate = self.samplerate[slide]
                samplesize = int(rate*150)
                index = np.random.choice(np.arange(len(imgpath)),size=samplesize,replace=False)

                #index = np.random.choice(np.arange(len(imgpath)),size=300,replace=False)
                imgpath = imgpath[index]
                
                label = np.zeros((len(imgpath),3))
                #label[:,self.slidefc[slide.split('-')[0]]]=1
                label[:,self.slidefc[slide][0]]=1

                #self.itmes.
                
                
                self.imgs.extend(imgpath.tolist())
                self.labels.extend(label.tolist())

                # 这样搞图像和标签都不对上了。。。
                #random.shuffle(self.imgs)
                #random.shuffle(self.labels)

        else:
            for slide in self._slide_list:
                patchdir = os.path.join('patchdata',slide)
                imgpath = np.load(os.path.join(patchdir,f'lv{self.level}_uniform_imglist.npy'))

                rate = self.samplerate[slide]
                samplesize = int(rate*150)
                index = np.random.choice(np.arange(len(imgpath)),size=samplesize,replace=False)

                imgpath = imgpath[index]

                label = np.zeros((len(imgpath),3))
                label[:,self.slidefc[slide][0]]=1

                
                
                self.imgs.extend(imgpath.tolist())
                self.labels.extend(label.tolist())

                #random.shuffle(self.imgs)
                #random.shuffle(self.labels)
        
        # shuffle data
        index = np.arange(len(self.imgs))
        random.shuffle(index)
        self.imgs = np.array(self.imgs)[index]
        self.labels = np.array(self.labels)[index]


        self._num_images = len(self.imgs)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path = self.imgs[idx]
        label = self.labels[idx]

        # label = 1: tumor
        #label = np.array(label)
        ## convert to one-hot
        #labeloh = np.zeros((3,))
        #labeloh[label] = 1
        #label = labeloh.astype(float)

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

class tumorImageDataset(Dataset):

    def __init__(self,slide_list,img_size,
                 crop_size=224, normalize=True):
        #self._data_path = data_path
        self._slide_list = slide_list
        self._img_size = img_size
        self._crop_size = crop_size
        self._normalize = normalize
        # 作色调、饱和度等调整
        #self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
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
        #img = self._color_jitter(img)

        # use left_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # use rotate
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)

        # PIL image: H W C
        # torch image: C H W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        img = img/255.0
        if self._normalize:
            img = (img - 128.0) / 128.0

        return img, label


class gradewsiImageDataset(Dataset):

    def __init__(self,slide_list,img_size,
                 crop_size=224, normalize=True):
        #self._data_path = data_path
        self._slide_list = slide_list
        self._img_size = img_size
        self._crop_size = crop_size
        self._normalize = normalize
        # 作色调、饱和度等调整
        #self._color_jitter = transforms.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
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
        #img = self._color_jitter(img)

        # use left_right flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # use rotate
        num_rotate = np.random.randint(0, 4)
        img = img.rotate(90 * num_rotate)

        # PIL image: H W C
        # torch image: C H W
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        img = img/255.0
        if self._normalize:
            img = (img - 128.0) / 128.0

        return img, label