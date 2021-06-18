import openslide
import numpy as np
import cv2
from PIL import Image
import json
import os
from tqdm import tqdm
from torchvision import transforms


ORIPATH = '/data/images/pathology/temp/Qingdao'
wsipath = 'wsidata_aug_uniform'

ratepath = 'rate_lv8.json'
with open(ratepath,'r') as f:
    rate = json.load(f)
patchsize = 512

if not os.path.exists(wsipath):
    os.makedirs(wsipath)
def getissue(path):
    level = 5
    slide = openslide.OpenSlide(path)
    tissue_mask = slide.read_region((0, 0), level, slide.level_dimensions[level])
    tissue_mask = cv2.cvtColor(np.array(tissue_mask), cv2.COLOR_RGBA2RGB)
    tissue_mask = cv2.cvtColor(tissue_mask, cv2.COLOR_BGR2HSV)
    tissue_mask = tissue_mask[:, :, 1]
    _, tissue_mask = cv2.threshold(tissue_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_mask = np.array(tissue_mask)
    return tissue_mask

def detect_tissue(tissue):
    tissue_mask = cv2.cvtColor(np.array(tissue), cv2.COLOR_RGBA2RGB)
    tissue_mask = cv2.cvtColor(tissue_mask, cv2.COLOR_BGR2HSV)
    tissue_mask = tissue_mask[:, :, 1]
    _, tissue_mask = cv2.threshold(tissue_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_mask = np.array(tissue_mask)
    if np.sum(tissue_mask>0)/(512*512)>0.1:
        return True
    else:
        return False

dirs = os.listdir('zipimgs')
wsilist = []
wsipatchlist = []
for s in tqdm(dirs):
    path = os.path.join(ORIPATH,s.split('.')[0],s.split('.')[0]+'.mrxs')
    slide = openslide.OpenSlide(path)
    #print(path)
    slidename = s.split('.')[0]
    scale = (int(patchsize/rate[slidename][0]),int(patchsize/rate[slidename][1]))
    tissue_mask = getissue(path)
    X_idcs, Y_idcs = np.where(tissue_mask)
    X = X_idcs*2**5
    Y = Y_idcs*2**5
    x1 = np.min(X)
    y1 = np.min(Y)
    x2 = np.max(X)
    y2 = np.max(Y)
    xm = (x1+x2)//2
    ym = (y1+y2)//2
    xs = [x1,x1,xm,xm]
    ys = [y1,ym,y1,ym]
    # 太小不能读取到全部组织区域
    for i in range(4):
        tissue = slide.read_region((ys[i], xs[i]),8,scale)
        tissue = cv2.resize(np.array(tissue),dsize=(patchsize,patchsize),interpolation=cv2.INTER_CUBIC)
        tissue = Image.fromarray(tissue)
        # 检测组织mask防止只切到很小一部分组织数据
        if detect_tissue(tissue):
            # data expansion
            tissue = cv2.cvtColor(np.array(tissue), cv2.COLOR_RGBA2RGB)
            tissue = Image.fromarray(tissue)

            img1 = transforms.RandomHorizontalFlip(p=1)(tissue)
            img2 = transforms.RandomVerticalFlip(p=1)(tissue)
        
            path0 = slidename + f'_{i}_0.png'
            path1 = slidename + f'_{i}_1.png'
            path2 = slidename + f'_{i}_2.png'
            
            tissue.save(os.path.join(wsipath, slidename + f'_{i}_0.png'))
            img1.save(os.path.join(wsipath, slidename + f'_{i}_1.png'))
            img2.save(os.path.join(wsipath, slidename + f'_{i}_2.png'))
            #img3.save(os.path.join(wsipath, slidename + '_3.png'))
            wsipatchlist.append(path0)
            wsipatchlist.append(path1)
            wsipatchlist.append(path2)

    wsilist.append(slidename)
np.save('wsidata_aug_uniform/wsilist.npy',wsilist)
np.save('wsidata_aug_uniform/wsipatchlist.npy',wsipatchlist)
print(len(wsipatchlist))
print('finish')
    