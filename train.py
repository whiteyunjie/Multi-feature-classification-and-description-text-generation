#coding=utf-8
import argparse
import collections
import os
import pickle
import json
import pandas as pd
import skimage.transform
from sklearn.metrics import accuracy_score,f1_score

os.environ["CUDA_VISIBLE_DEVICES"]="1,3,4,5"
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss
from tqdm import tqdm
#import cv2

from clsmodels import resnet_fpn
from clsmodels import senet_fpn
from clsmodels import resnet_fpn_sf
from clsmodels.resnet_fpn import Resnet_fpn_classifier,Resnet_fc_classifier
#from clsmodels.resnet_fpn_sf import Resnet_fpn_classifier,Resnet_fc_classifier
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.font_manager import FontProperties


from imagedataset import ImageDataset
#from logger import Logger

from sklearn.metrics import roc_curve, auc
from itertools import cycle

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(description='fpn classification')
#parser.add_argument('--mode',default='classification',
#                    help='classification or segmentation')
parser.add_argument('--savepath',default='res/models',
                    help='path to save models')
parser.add_argument('--lr',default=0.001,type = int,
                    help = 'learning rate')
# 多gpu训练
parser.add_argument("--gpu",default="0",type = str,
                    help="training gpu")
parser.add_argument("--epoch",default=20,type = int,
                    help="number of epoches of training")
parser.add_argument("--bs",default=32,type = int,
                    help="number of batchsize of training")
parser.add_argument("--model",default='resnet34_fpn_classifier',
                    help="model to be trained")
parser.add_argument("--rws",default='',
                    help="model weights")
parser.add_argument('--resume_epoch', type=int, default=-1)


args = parser.parse_args()

ALPHA = 1

slist = np.load('fin.npy')
'''
with open('sample_cellfeatures1.json','r') as f:
    cf = json.load(f)
slidelist1 = []
for slide in slist:
    if slide in cf:
        slidelist1.append(slide)

with open('cellfeatures.json','r') as f:
    cfall = json.load(f)
slidelist2 = []
i = 0
for slide in slist:
    if slide not in cf and slide in cfall and i < 5:
        i = i + 1
        slidelist2.append(slide)
'''
with open('cftrainnf.json','r') as f:
    cf = json.load(f)
slidelist1 = []
for slide in slist:
    if slide in cf:
        slidelist1.append(slide)
i = 0
with open('dataconfig/rateall.json','r') as f:
    cf = json.load(f)
slidelist1 = []
for slide in slist:
    if slide in cf:
        slidelist1.append(slide)

with open('cellfeatures3.json','r') as f:
    cf = json.load(f)
    slidelist = []
    for slide in slist:
        if slide in cf and slide not in slidelist1:
            i = i + 1 
            slidelist.append(slide)
        
with open('cftestnf.json','r') as f:
    cf2 = json.load(f)
slidelist2 = []
for slide in slist:
    if slide in cf2:
        slidelist2.append(slide)
#print(slidelist)
testslide = np.load('testslide.npy')

def train(args):
    checkpoints_dir = f'temp/checkpoints/{args.model}'
    results_dir = f'temp/results/{args.model}'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if args.model == 'resnet34_fpn_classifier':
        model = resnet_fpn.resnet34_fpn_classifier(num_classes=[2,2,2],pretrained = True)
    if args.model == 'resnet50_fpn_classifier':
        model = resnet_fpn.resnet50_fpn_classifier(num_classes=[2,2,2],pretrained = True)
    if args.model == 'resnet34_fc_classifier':
        model = resnet_fpn.resnet34_fc_classifier(num_classes=[2,2,2],pretrained = True)
    if args.model == 'se_resnet50_fpn_classifier':
        model = senet_fpn.se_resnet50_fpn_classifier(num_classes=[2,2,2])

    if args.rws!= '':
        print('load model from', args.rws)
        model = torch.load(rws).cuda()
    else:
        model = model.cuda()

    model = torch.nn.DataParallel(model).cuda()

    #retinanet = torch.nn.DataParallel(retinanet).cuda()
    with open('dataconfig/rateall.json','r') as f:
        samplerate = json.load(f)
    traindataset = ImageDataset(slide_list = slidelist1,
                                img_size = 512,
                                level = 1,
                                samplerate = samplerate,
                                is_training = True)
    traindataloader = DataLoader(traindataset,
                                batch_size = args.bs,
                                shuffle = True,
                                drop_last = True)
    
    with open('dataconfig/rateval_all.json','r') as f:
            samplerate = json.load(f)
    # 补充验证集，同时修改dataset
    valdataset = ImageDataset(slide_list = testslide,
                                img_size = 512,
                                level = 1,
                                samplerate = samplerate,
                                is_training = False)
    valdataloader = DataLoader(valdataset,
                                batch_size = args.bs,
                                shuffle = True,
                                drop_last = True)

    # training prepare
    model.training = True
    #optimizer = optim.Adam(model.parameters(),lr=args.lr)
    # multi-optim
    '''
    optimizer = optim.SGD([
            {'params': model.module.classificationModel1.parameters(), 'lr': 1e-6},
            {'params': model.module.classificationModel2.parameters(), 'lr': 1e-5},
            {'params': model.module.classificationModel3.parameters(), 'lr': 1e-5},
        ], lr=1e-5, momentum=0.9,weight_decay=1e-3)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True, factor=0.1)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30])
    '''
    


    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-3)
    #optimizer = torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay = 1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,verbose=True)
    #loss_fn = CrossEntropyLoss().cuda()
    loss_fn = BCEWithLogitsLoss().cuda()
    #loss_fn = BCELoss().cuda()

    model.train()
    bestacc1 = 0
    countepoch1 = 0

    bestacc2 = 0
    countepoch2 = 0

    bestacc3 = 0
    countepoch3 = 0
    #model.module.freeze_bn()

    print('Num training images: {}'.format(len(traindataset)))
    epochs = 200
    results = {'loss_nf':[],'loss_ncr':[],'loss_shape':[],'loss':[],
                'loss_nf_val':[],'loss_ncr_val':[],'loss_shape_val':[],'loss_val':[],
                'acc_nf':[],'acc_ncr':[],'acc_shape':[],
                'acc_nf_val':[],'acc_ncr_val':[],'acc_shape_val':[]}
    for epoch in range(args.resume_epoch+1,epochs):
        model.train()
        #model.module.freeze_bn()
        epoch_loss = []
        loss_cls1_hist = []
        loss_cls2_hist = []
        loss_cls3_hist = []
        acc_cls1_hist = []
        acc_cls2_hist = []
        acc_cls3_hist = []

        #with torch.set_grad_enabled(True):
        progress_bar = tqdm(enumerate(traindataloader),total = len(traindataloader))
        for iter_num,data in progress_bar:
            optimizer.zero_grad()
            img, label_nf, label_ncr, label_shape = data
            #print(img.shape)
            #print(label_nf.shape)
            #print(label_shape.shape)
            img = img.cuda().float()
            label_nf = label_nf.cuda().float()
            label_ncr = label_ncr.cuda().float()
            label_shape = label_shape.cuda().float()
            output1,output2,output3 = model(img)

            #print(output1.min())
            #print(output1.max())
            #print(output2.min())
            #print(output2.max())
            #print(output1)
            output1 = torch.squeeze(output1)
            output2 = torch.squeeze(output2)
            output3 = torch.squeeze(output3)
            #print(output1.size())
            #print(output1)

            #loss_nf = loss_fn(output1,label_nf.argmax(axis=1))
            #loss_shape = loss_fn(output2,label_shape.argmax(axis=1))

            loss_nf = loss_fn(output1,label_nf)
            loss_ncr = loss_fn(output2,label_ncr)
            loss_shape = loss_fn(output3,label_shape)

            loss = loss_nf + 1.2*loss_ncr +  ALPHA*loss_shape

            #梯度阶段，防止梯度消失
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

            loss.backward()
            optimizer.step()

            #prob = output.sigmoid()
            pred_nf = output1.argmax(axis=1)
            pred_ncr = output2.argmax(axis=1)
            pred_shape = output3.argmax(axis=1)
            #print(pred_shape)
            #print(label_shape.argmax(axis=1))

            #print(pred_nf)
            #print(label_nf.argmax(axis=1))
            acc_nf = (pred_nf == label_nf.argmax(axis=1)).sum().data * 1.0/args.bs
            acc_ncr = (pred_ncr == label_ncr.argmax(axis=1)).sum().data * 1.0/args.bs
            acc_shape = (pred_shape == label_shape.argmax(axis=1)).sum().data * 1.0/args.bs

            loss_cls1_hist.append(float(loss_nf))
            loss_cls2_hist.append(float(loss_ncr))
            loss_cls3_hist.append(float(loss_shape))
            epoch_loss.append(float(loss))
            acc_cls1_hist.append(float(acc_nf))
            acc_cls2_hist.append(float(acc_ncr))
            acc_cls3_hist.append(float(acc_shape))


            progress_bar.set_description(
                f'epoch: {epoch} cls_nf: {np.mean(loss_cls1_hist):1.4f} cls_ncr: {np.mean(loss_cls2_hist):1.4f} cls_shape: {np.mean(loss_cls3_hist):1.4f} loss: {np.mean(epoch_loss):1.4f} acc_nf: {np.mean(acc_cls1_hist):1.4f} acc_ncr:{np.mean(acc_cls2_hist):1.4f} acc_shape: {np.mean(acc_cls3_hist):1.4f}')
            
            del loss_nf
            del loss_shape
        results['loss_nf'].append(np.mean(loss_cls1_hist))
        results['loss_ncr'].append(np.mean(loss_cls2_hist))
        results['loss_shape'].append(np.mean(loss_cls3_hist))
        results['loss'].append(np.mean(epoch_loss))
        results['acc_nf'].append(np.mean(acc_cls1_hist))
        results['acc_ncr'].append(np.mean(acc_cls2_hist))
        results['acc_shape'].append(np.mean(acc_cls3_hist))
    
        # validation
        with torch.no_grad():
            model.eval()
            epoch_loss_val = []
            loss_cls1_val = []
            loss_cls2_val = []
            loss_cls3_val = []
            acc_cls1_val = []
            acc_cls2_val = []
            acc_cls3_val = []

            progress_bar = tqdm(enumerate(valdataloader),total = len(valdataloader))
            for iter_num,data in progress_bar:
                img, label_nf, label_ncr, label_shape = data
                img = img.cuda().float()
                label_nf = label_nf.cuda().float()
                label_ncr = label_ncr.cuda().float()
                label_shape = label_shape.cuda().float()
                output1,output2,output3 = model(img)

                output1 = torch.squeeze(output1)
                output2 = torch.squeeze(output2)
                output3 = torch.squeeze(output3)

                loss_nf = loss_fn(output1,label_nf)
                loss_ncr = loss_fn(output2,label_ncr)
                loss_shape = loss_fn(output3,label_shape)

                loss = loss_nf + loss_ncr +  ALPHA*loss_shape

                #loss_nf = loss_fn(output1,label_nf.argmax(axis=1))
                #loss_shape = loss_fn(output2,label_shape.argmax(axis=1))

                #loss = loss_nf + ALPHA*loss_shape

                pred_nf = output1.argmax(axis=1)
                pred_ncr = output2.argmax(axis=1)
                pred_shape = output3.argmax(axis=1)

                #print(output1)
                #print(pred_shape)
                #print(label_shape.argmax(axis=1))

                #print(pred_nf)
                #print(label_nf.argmax(axis=1))
                
                acc_nf = (pred_nf == label_nf.argmax(axis=1)).sum().data * 1.0/args.bs
                acc_ncr = (pred_ncr == label_ncr.argmax(axis=1)).sum().data * 1.0/args.bs
                acc_shape = (pred_shape == label_shape.argmax(axis=1)).sum().data * 1.0/args.bs

                loss_cls1_val.append(float(loss_nf))
                loss_cls2_val.append(float(loss_ncr))
                loss_cls3_val.append(float(loss_shape))
                epoch_loss_val.append(float(loss))
                acc_cls1_val.append(float(acc_nf))
                acc_cls2_val.append(float(acc_ncr))
                acc_cls3_val.append(float(acc_shape))

                progress_bar.set_description(
                    f'val: epoch: {epoch} cls_nf: {np.mean(loss_cls1_val):1.4f} cls_ncr: {np.mean(loss_cls2_val):1.4f} cls_shape: {np.mean(loss_cls3_val):1.4f} loss: {np.mean(epoch_loss_val):1.4f} acc_nf: {np.mean(acc_cls1_val):1.4f} acc_ncr: {np.mean(acc_cls2_val):1.4f} acc_shape: {np.mean(acc_cls3_val):1.4f}')
                
                del loss_nf
                del loss_shape
            
            results['loss_nf_val'].append(np.mean(loss_cls1_val))
            results['loss_ncr_val'].append(np.mean(loss_cls2_val))
            results['loss_shape_val'].append(np.mean(loss_cls3_val))
            results['loss_val'].append(np.mean(epoch_loss_val))
            results['acc_nf_val'].append(np.mean(acc_cls1_val))
            results['acc_ncr_val'].append(np.mean(acc_cls2_val))
            results['acc_shape_val'].append(np.mean(acc_cls3_val))
        
        #break
        scheduler.step(np.mean(epoch_loss_val))

        if bestacc1 < np.mean(acc_cls1_val):
            torch.save(model.state_dict(),f'temp/checkpoints/{args.model}/{args.model}_{epoch}_nf_1_best.pth.tar')
            bestacc1 = np.mean(acc_cls1_val)
            countepoch1 = 0
        else:
            countepoch1 = countepoch1 + 1
            if countepoch1 > 10:
                print('early stopping(nf)')
        
        if bestacc2 < np.mean(acc_cls2_val):
            torch.save(model.state_dict(),f'temp/checkpoints/{args.model}/{args.model}_{epoch}_ncr_1_best.pth.tar')
            bestacc2 = np.mean(acc_cls2_val)
            countepoch2 = 0
        else:
            countepoch2 = countepoch2 + 1
            if countepoch2 > 10:
                print('early stopping(ncr)')

        if bestacc3 < np.mean(acc_cls3_val):
            torch.save(model.state_dict(),f'temp/checkpoints/{args.model}/{args.model}_{epoch}_shape_1_best.pth.tar')
            bestacc3 = np.mean(acc_cls3_val)
            countepoch3 = 0
        else:
            countepoch3 = countepoch3 + 1
            if countepoch3 > 10:
                print('early stopping(shape)')
                #break
        #model.eval()
        if (epoch+1)%5 == 0:
            torch.save(model.state_dict(),f'temp/checkpoints/{args.model}/{args.model}_{epoch}_1.pth.tar')
        with open(f'temp/results/{args.model}/re_{args.resume_epoch}_{args.resume_epoch+epochs}_1_0406.pkl','wb') as f:
            pickle.dump(results,f)

def main():
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()