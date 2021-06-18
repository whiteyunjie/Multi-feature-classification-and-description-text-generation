import sys
import os
import argparse
import logging
import json
import time

import numpy as np
import torch
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="1,3"
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, BCELoss ,CrossEntropyLoss
from torch.optim import SGD
from torchvision import models
from torch import nn

from tqdm import tqdm
#from pytorchtools import EarlyStopping
from tensorboardX import SummaryWriter
from imagedataset import WSIimageDataset, GradeImageDataset, ImageDataset
from clsmodels.models import create_model

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(description='classification')
#parser.add_argument('--mode',default='classification',
#                    help='classification or segmentation')
parser.add_argument('--savepath',default='res/clsmodels',
                    help='path to save models')
parser.add_argument('--lr',default=1e-4,type = int,
                    help = 'learning rate')
# 多gpu训练
parser.add_argument("--gpu",default="0",type = str,
                    help="training gpu")
parser.add_argument("--epoch",default=200,type = int,
                    help="number of epoches of training")
parser.add_argument("--bs",default=32,type = int,
                    help="number of epoches of training")
parser.add_argument("--method",default='cnn',
                    help="method:cnn,SVM,logistic,random forest")
parser.add_argument("--weights",default='',
                    help="model weights")



args = parser.parse_args()

## 验证集有问题，重新生成
slist = np.load('fin.npy')
with open('cellfeatures2.json','r') as f:
    cf = json.load(f)
slidelist = []
for slide in slist:
    if slide in cf:
        slidelist.append(slide)

i = 0
slist = np.load('fin.npy')
with open('cftrainnf.json','r') as f:
    cf = json.load(f)
slidelist1 = []
for slide in slist:
    if slide in cf:
        slidelist1.append(slide)
        i = i + 1
i = 0
i = 0
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
        i = i + 1

        
index = np.array([1,3,10,20])
slidelistval = np.array(slidelist)[index]
slidelisttest = slidelistval.tolist() + slidelist2    
def train(args):
    classnum = 0
    if args.mission == 'envelop' or args.mission == 'necrosis':
        traindataset = WSIimageDataset(is_training = True,
                                        mission = args.mission)
        # num_workers
        traindataloader = DataLoader(traindataset,batch_size = args.bs)
        valdataset = WSIimageDataset(is_training = False,
                                        mission = args.mission)
        # num_workers
        valdataloader = DataLoader(traindataset,batch_size = args.bs)
        classnum = 2
    elif args.mission == 'grade':
        traindataset = GradeImageDataset(slidelist1,256,2,True)                            
        traindataloader = DataLoader(traindataset,batch_size = args.bs)

        valdataset = GradeImageDataset(slidelist2,256,2,True)                            
        valdataloader = DataLoader(valdataset,batch_size = args.bs)
        classnum = 3
    elif args.mission == 'nf' or args.mission == 'ncr':
        #print(slidelist)
        traindataset = ImageDataset(slidelist1,512,1,True)                            
        traindataloader = DataLoader(traindataset,batch_size = args.bs)

        valdataset = ImageDataset(slidelisttest,512,1,False)                            
        valdataloader = DataLoader(valdataset,batch_size = args.bs)
        classnum = 2
    elif args.mission == 'shape':
        traindataset = ImageDataset(slidelist1,512,1,True)                            
        traindataloader = DataLoader(traindataset,batch_size = args.bs)

        valdataset = ImageDataset(slidelist2,512,1,True)                            
        valdataloader = DataLoader(valdataset,batch_size = args.bs)
        classnum = 2
    else:
        raise ValueError(f'unsupported misson name {args.mission}')

    checkpoints_dir = f'temp/checkpoints/{args.mission}/{args.backbone}'
    results_dir = f'temp/results/{args.mission}/{args.backbone}'
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    model = create_model(args.backbone,pretrained = True,num_classes=classnum)
    if args.weights != '':
        print('load model from ',args.weights)
        model.load_state_dict(torch.load(args,weights))
    #研究一下dataparallel
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    #loss_fn = CrossEntropyLoss().cuda()
    loss_fn = BCEWithLogitsLoss().cuda()
    #optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay = 5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10,verbose=True)

    if args.mission == 'nf' or args.mission == 'ncr':
        #early_stopping = EarlyStopping(patience=10, verbose=True)
        bestacc = 0

        summary_train = {'epoch': 0, 'step': 0}
        #summary_valid = {'loss': float('inf'), 'acc': 0}
        summary_writer = SummaryWriter('res/models')
        loss_valid_best = float('inf')
        i = 0

        model.train()
        print('Num training images: {}'.format(len(traindataset)))
        results = {'loss':[],'loss_val':[],'acc':[],'acc_val':[]}
        for epoch in range(args.epoch):
            model.train()
            epoch_loss = []
            epoch_acc = []
            progress_bar = tqdm(traindataloader)
            with torch.set_grad_enabled(True):
                for imgs,label1,label2,label3 in progress_bar:
                    #i = i + 1
                    #print(imgs.type)
                    #print(label.type)

                    imgs = imgs.float().cuda()
                    label = label2.float().cuda()

                    output = model(imgs)
                    #print(output)
                    output = torch.squeeze(output)
                    loss = loss_fn(output,label)
                    #loss = loss_fn(output,label.argmax(axis=1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pred = output.argmax(axis=1)
                    #print(label.argmax(axis=1))
                    #print(pred)
                    acc_data = (pred == label.argmax(axis=1)).sum().data * 1.0/args.bs
                    loss_data = loss.data

                    epoch_loss.append(float(loss_data))
                    epoch_acc.append(float(acc_data))

                    progress_bar.set_description(
                        f'epoch: {epoch} loss: {np.mean(epoch_loss):1.4f} acc_all: {np.mean(epoch_acc):1.4f}')
                            
                    #del loss



                results['loss'].append(np.mean(epoch_loss))
                results['acc'].append(np.mean(epoch_acc))

            with torch.no_grad():
                model.eval()
                epoch_loss_val = []
                epoch_acc_val = []

                progress_bar = tqdm(valdataloader)
                for imgs,label1,label2,label3 in progress_bar:
                    imgs = imgs.float().cuda()
                    label = label2.float().cuda()

                    output = model(imgs)

                    output = torch.squeeze(output)
                    #loss = loss_fn(output,label.argmax(axis=1))
                    loss = loss_fn(output,label)

                    prob = output.sigmoid()
                    pred = output.argmax(axis=1)

                    #print(label.argmax(axis=1))
                    #print(pred)

                    acc_data = (pred == label.argmax(axis=1)).sum().data * 1.0/args.bs
                    loss_data = loss.data

                    epoch_loss_val.append(float(loss_data))
                    epoch_acc_val.append(float(acc_data))

                    progress_bar.set_description(
                        f'val:  epoch: {epoch} loss: {np.mean(epoch_loss_val):1.4f} acc_all: {np.mean(epoch_acc_val):1.4f}')
                            
                    del loss
                results['loss_val'].append(np.mean(epoch_loss_val))
                results['acc_val'].append(np.mean(epoch_acc_val))

            if bestacc < np.mean(epoch_acc_val):
                torch.save(model.state_dict(),f'temp/checkpoints/{args.mission}/{args.backbone}/{args.backbone}_{epoch}_1_best.pth.tar')
                bestacc = np.mean(epoch_acc_val)
                countepoch = 0
            else:
                countepoch = countepoch + 1
                if countepoch > 10:
                    print('early stopping')
                    break

            scheduler.step(np.mean(epoch_loss_val))
            
            model.eval()
            if (epoch+1)%5 == 0:
                torch.save(model.state_dict(),f'temp/checkpoints/{args.mission}/{args.backbone}/{args.backbone}_{epoch}_1.pth.tar')
            with open(f'temp/results/{args.mission}/{args.backbone}/re_{epoch}.pkl','wb') as f:
                pickle.dump(results,f)

    if args.mission == 'envelop' or args.mission == 'necrosis':
        #early_stopping = EarlyStopping(patience=10, verbose=True)
        bestacc = 0

        summary_train = {'epoch': 0, 'step': 0}
        #summary_valid = {'loss': float('inf'), 'acc': 0}
        summary_writer = SummaryWriter('res/models')
        loss_valid_best = float('inf')
        i = 0

        model.train()
        print('Num training images: {}'.format(len(traindataset)))
        results = {'loss':[],'loss_val':[],'acc':[],'acc_val':[]}
        for epoch in range(args.epoch):
            model.train()
            epoch_loss = []
            epoch_acc = []
            progress_bar = tqdm(traindataloader)
            with torch.set_grad_enabled(True):
                for imgs,label in progress_bar:

                    imgs = imgs.float().cuda()
                    label = label.float().cuda()

                    output = model(imgs)
                    #print(output)
                    output = torch.squeeze(output)
                    loss = loss_fn(output,label)
                    #loss = loss_fn(output,label.argmax(axis=1))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    pred = output.argmax(axis=1)
                    #print(label.argmax(axis=1))
                    #print(pred)
                    acc_data = (pred == label.argmax(axis=1)).sum().data * 1.0/args.bs
                    loss_data = loss.data

                    epoch_loss.append(float(loss_data))
                    epoch_acc.append(float(acc_data))

                    progress_bar.set_description(
                        f'epoch: {epoch} loss: {np.mean(epoch_loss):1.4f} acc_all: {np.mean(epoch_acc):1.4f}')
                            
                    #del loss



                results['loss'].append(np.mean(epoch_loss))
                results['acc'].append(np.mean(epoch_acc))

            with torch.no_grad():
                model.eval()
                epoch_loss_val = []
                epoch_acc_val = []

                progress_bar = tqdm(valdataloader)
                for imgs,label in progress_bar:
                    imgs = imgs.float().cuda()
                    label = label.float().cuda()

                    output = model(imgs)

                    output = torch.squeeze(output)
                    #loss = loss_fn(output,label.argmax(axis=1))
                    loss = loss_fn(output,label)

                    prob = output.sigmoid()
                    pred = output.argmax(axis=1)

                    #print(label.argmax(axis=1))
                    #print(pred)

                    acc_data = (pred == label.argmax(axis=1)).sum().data * 1.0/args.bs
                    loss_data = loss.data

                    epoch_loss_val.append(float(loss_data))
                    epoch_acc_val.append(float(acc_data))

                    progress_bar.set_description(
                        f'val:  epoch: {epoch} loss: {np.mean(epoch_loss_val):1.4f} acc_all: {np.mean(epoch_acc_val):1.4f}')
                            
                    del loss
                results['loss_val'].append(np.mean(epoch_loss_val))
                results['acc_val'].append(np.mean(epoch_acc_val))

            if bestacc < np.mean(epoch_acc_val):
                torch.save(model.state_dict(),f'temp/checkpoints/{args.mission}/{args.backbone}/{args.backbone}_{epoch}_best.pth.tar')
                bestacc = np.mean(epoch_acc_val)
                countepoch = 0
            else:
                countepoch = countepoch + 1
                if countepoch > 10:
                    print('early stopping')
                    break

            scheduler.step(np.mean(epoch_loss_val))
            
            model.eval()
            if (epoch+1)%5 == 0:
                torch.save(model.state_dict(),f'temp/checkpoints/{args.mission}/{args.backbone}/{args.backbone}_{epoch}.pth.tar')
            with open(f'temp/results/{args.mission}/{args.backbone}/re_{epoch}.pkl','wb') as f:
                pickle.dump(results,f)



def main():
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()