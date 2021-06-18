import numpy as np
import openslide
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
import json
import copy
import cv2
import os

level = 5
center = "/data/images/pathology/temp/Qingdao/"

slidelist = []
spathlist = os.listdir('results')
for spath in spathlist:
    if spath.split('.')[1] == 'xml':
        slidelist.append(spath.split('.')[0])
print(len(slidelist))

for fp in tqdm(slidelist):
    tumormask_path = 'patchdata' + '/' + fp
    savepath = tumormask_path + '/' + 'mask.npy'
    if os.path.exists(savepath) and fp != '17-043414-4':
        continue
    filepath = center + fp + '/' + fp +'.mrxs'
    #print(filepath)
    # generate certain number and level patches
    slide = openslide.OpenSlide(filepath)
    # extract tissue region
    tissue_mask = slide.read_region((0, 0), 5, slide.level_dimensions[5])
    tissue_mask = cv2.cvtColor(np.array(tissue_mask), cv2.COLOR_RGBA2RGB)
    tissue_mask = cv2.cvtColor(tissue_mask, cv2.COLOR_BGR2HSV)
    tissue_mask = tissue_mask[:, :, 1]
    _, tissue_mask = cv2.threshold(tissue_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue_mask = np.array(tissue_mask)
    # read tumor region xml
    ann_path = os.path.join('results',fp+'.xml')
    root = ET.parse(ann_path).getroot()
    annotations_tumor = root.findall('./Annotations/Annotation[@PartOfGroup="PLC"]')
    json_dict = {}
    json_dict['tumor'] = []
    for annotation in annotations_tumor:
        X = list(map(lambda x: float(x.get('X')),
                annotation.findall('./Coordinates/Coordinate')))
        Y = list(map(lambda x: float(x.get('Y')),
                annotation.findall('./Coordinates/Coordinate')))
        vertices = np.round([X, Y]).astype(int).transpose().tolist()
        if len(vertices) >=10:
            name = annotation.attrib['Name']
            json_dict['tumor'].append({'name': name, 'vertices': vertices})
    tumor_polygons = json_dict['tumor']

    w,h = slide.level_dimensions[5]
    tumor_mask = np.zeros((h,w))
    for tumor_polygon in tumor_polygons:
        #name = tumor_polygon['name']
        vertices = np.array(tumor_polygon['vertices']) / 2**5
        vertices = vertices.astype(np.int32)
        cv2.fillPoly(tumor_mask, [vertices], (255))
    tumor_mask = tumor_mask[:] > 127
    tumor_mask = np.transpose(tumor_mask)
    ## 这一部分要单独写出来保存成npy文件，方便后面直接生成采样点

    tumormask_path = 'patchdata' + '/' + fp
    if not os.path.exists(tumormask_path):
        os.makedirs(tumormask_path)
    savepath = tumormask_path + '/' + 'mask.npy'
    print(fp)
    np.save(savepath,tumor_mask)
    #tumor_mask = tumor_mask & self.tissue_mask
    