from tqdm import tqdm
import os
from mask import Slidemask
from sklearn.model_selection import train_test_split
import numpy as np
import openslide


numpatch = 300
levels = [0]
#levels = [5,4] #暂时不考虑6
center = "/data/images/pathology/temp/Qingdao/"

slidelist = []
spathlist = os.listdir('results')
for spath in spathlist:
    if spath.split('.')[1] == 'xml':
        slidelist.append(spath.split('.')[0])

for fp in slidelist:
    #fp = '201603916-1'
    filepath = center + fp + '/' + fp +'.mrxs'
    print(filepath)

    # generate certain number and level patches
    ann_path = os.path.join('patchdata',fp,'mask.npy')
    tumor_mask = np.load(ann_path)
    slide = openslide.OpenSlide(filepath)
    X_idcs, Y_idcs = np.where(tumor_mask)
    center_points = np.stack(np.vstack((X_idcs.T, Y_idcs.T)), axis=1)

    if center_points.shape[0] > numpatch:
        sampled_points = center_points[np.random.randint(center_points.shape[0],
                                                        size=numpatch), :]
    else:
        sampled_points = center_points

    sampled_points = (sampled_points * 2 ** 5).astype(np.int32)
    #print(sampled_points)


    # save patch images
    if len(sampled_points)==0:
        print('no tumor regions')
    else:
        for levelscale in levels:
            imglist = []
            print(f'generating tumor patch ,level({levelscale})  :')
            path = 'patchdata' + '/' +  fp + '/' + str(levelscale)
            if not os.path.exists(path):
                os.makedirs(path)
            for i in tqdm(range(len(sampled_points))):
                xc,yc = sampled_points[i]
                #x = int(int(xc) - 512 / 2)
                #y = int(int(yc) - 512 / 2)
                x = int(int(xc) - 256 / 2)
                y = int(int(yc) - 256 / 2)
                # wsi_path = os.path.join('z56a/', pid + '.tif')
                # slide_mask = openslide.OpenSlide(wsi_path)
                img = slide.read_region((x, y), levelscale,(256, 256)).convert('RGB')

                imglist.append(os.path.join(path, str(i) + '.png'))
                img.save(os.path.join(path, str(i) + '.png'))
            
            imgarray = np.array(imglist)
            y = np.zeros(imgarray.shape)
            train,val,_,_ = train_test_split(imgarray,y,train_size=0.8,random_state=14)
            #trainlist = train.tolist()
            #vallist = val.tolist()
            np.save('patchdata' + '/' + fp+'/'+ f'lv{levelscale}' + '_' +'imglist.npy',imglist)
            np.save('patchdata' + '/' + fp+'/'+ f'lv{levelscale}' + '_' +'trainlist.npy',train)
            np.save('patchdata' + '/' + fp+'/'+ f'lv{levelscale}' + '_' +'vallist.npy',val)
            #np.save('data/slide.npy',slidelist)
    #break            
print('finish')
