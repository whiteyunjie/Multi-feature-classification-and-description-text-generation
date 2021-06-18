## metric
## visual results
## 1.acc、loss curve by epoch
## 2.test model,get acc、loss、confusion matrix
##   try notebook
## 3.test model,generate whole-slide-level description and visual results

import sys
import os
import argparse
import logging
import json
import time

import numpy as np
import torch
import pickle
os.environ["CUDA_VISIBLE_DEVICES"]="0,4,5"
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, BCELoss ,CrossEntropyLoss
from torch.optim import SGD
from torchvision import models
from torch import nn

from tqdm import tqdm
from tensorboardX import SummaryWriter
from imagedataset import ImageDataset, WSIimageDataset, GradeImageDataset, tumorImageDataset
from testloader import WSIimageDatasetpredict
from clsmodels.models import create_model
from clsmodels import resnet_fpn
from sklearn.metrics import roc_curve, auc, confusion_matrix
import itertools
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='metric')
parser.add_argument("--mission",default='nf',
                    help='metric mission type: tumor,nf,necrosis,envelop,grade,wsi')
parser.add_argument("--model",default='se_resnext50_32x4d',
                    help="model to be tested")
parser.add_argument("--weights",default='',
                    help="model weights")

args = parser.parse_args()


def compute_pr_re(y_true,y_pred,title):
    cm = confusion_matrix(y_true,y_pred,labels = [1,0])
    TP = cm[0,0]
    FP = cm[1,0]
    FN = cm[0,1]
    TN = cm[1,1]

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    specifity = TN/(TN+FP)
    accuracy = (TP+TN)/(TP+FP+FN+TN)

    return precision,recall,specifity,accuracy

def plot_roc(label,prob,title):
    fpr, tpr, threshold = roc_curve(label,prob)
    roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange',
            lw=1, label='AUC = %0.2f' % roc_auc)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    #plt.show()


def plot_confusion_matrix(cm,labels_name,title='confusion matrix'):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if float(num) > thresh else "black")
    
    plt.ylabel('True label')    
    plt.xlabel('Predicted label')

def test(args):
    if args.mission == 'tumor':
        # dataset
        slidelist = np.load('data/slide.npy')
        testdataset = tumorImageDataset(slidelist,256,224,False)
        testdataloader = DataLoader(traindataset,batch_size = 32)
        # load model
        model = create_model(args.model, pretrained=True, num_classes=1, load_backbone_weights=False)
        model_file = 'PLC/trainable_0_3000/best_pretrained.pth'
        print(f'loading {model_file}...')
        model.load_state_dict(torch.load(model_file))
        model = nn.DataParallel(model)
        model = model.cuda()
        
        loss_fn = BCEWithLogitsLoss().cuda()

        model.eval()

        acclist = []
        losslist = []
        predlist = []
        problist = []

        with torch.no_grad():
            progress_bar = tqdm(testdataloader)
            for imgs,label in progress_bar:
                imgs = imgs.float().cuda()
                label = label.float().cuda()

                output = model(imgs)
                output = torch.squeeze(output)
                loss = loss_fn(output,label)

                prob = output.sigmoid()
                pred = (prob>=0.5).type(torch.cuda.FloatTensor)
                
                acc_data = (pred == label).sum().data * 1.0/32
                loss_data = loss.data

                acclist.append(float(acc_data))
                losslist.append(float(loss_data))
                predlist.extend(pred)
        
                progress_bar.set_description(f'loss: {np.mean(losslist):1.4f} acc: {np.mean(acclist):1.4f}')
        
        fpr, tpr, threshold = roc_curve(label.cpu(),prob.cpu())
        roc_auc = auc(fpr,tpr)
        plt.figure(figsize=(8, 5))
        plt.plot(fpr, tpr, color='darkorange',
                lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC(tumor classification)')
        plt.legend(loc="lower right")
        plt.savefig('results/images/tumor/roc.png')
        #plt.show()

    if args.mission == 'nf':
        slist = np.load('fin.npy')
        with open('sample_cellfeatures1.json','r') as f:
            cf = json.load(f)
        with open('cellfeatures.json','r') as f:
            cf_all = json.load(f)
        slidelist = []
        i = 0
        for slide in slist:
            if slide not in cf and slide in cf_all and i < 10:
                slidelist.append(slide)
                i = i + 1
        
        model = resnet_fpn.resnet34_fpn_classifier(num_classes=[4,2],pretrained = True)
        model.load_state_dict(torch.load('temp/checkpoints/resnet_fpn_classifier/resnet_fpn_classifier_49_3.pth.tar'))

        model = model.cuda()

        #model = torch.nn.DataParallel(model).cuda()

        #retinanet = torch.nn.DataParallel(retinanet).cuda()
        testdataset = ImageDataset(slide_list = slidelist,
                                    img_size = 512,
                                    level = 3,
                                    is_training = True)
        testdataloader = DataLoader(testdataset,
                                    batch_size = 32,
                                    shuffle = True,
                                    drop_last = True)
        loss_fn = CrossEntropyLoss().cuda()
        #loss_fn = BCELoss().cuda()

        model.eval()
        #model.module.freeze_bn()

        print('Num training images: {}'.format(len(traindataset)))
        model.eval()
        epoch_loss_val = []
        loss_cls1_val = []
        loss_cls2_val = []
        acc_cls1_val = []
        acc_cls2_val = []
        pred1 = []
        pred2 = []
        true1 = []
        true2 = []
        prob = []

        progress_bar = tqdm(enumerate(testdataloader),total = len(testdataloader))
        for iter_num,data in progress_bar:
            img, label_nf, label_shape = data
            img = img.cuda().float()
            label_nf = label_nf.cuda().float()
            label_shape = label_shape.cuda().float()
            output1,output2 = model(img)

            output1 = torch.squeeze(output1)
            output2 = torch.squeeze(output2)

            loss_nf = loss_fn(output1,label_nf.argmax(axis=1))
            loss_shape = loss_fn(output2,label_shape.argmax(axis=1))

            loss = loss_nf + ALPHA*loss_shape

            pred_nf = output1.argmax(axis=1)
            pred_shape = output2.argmax(axis=1)

            prob2 = output2.sigmoid()

            #print(pred_shape)
            #print(label_shape.argmax(axis=1))
            
            acc_nf = (pred_nf == label_nf.argmax(axis=1)).sum().data * 1.0/32
            acc_shape = (pred_shape == label_shape.argmax(axis=1)).sum().data * 1.0/32

            loss_cls1_val.append(float(loss_nf))
            loss_cls2_val.append(float(loss_shape))
            epoch_loss_val.append(float(loss))
            acc_cls1_val.append(float(acc_nf))
            acc_cls2_val.append(float(acc_shape))
            pred1.extend(np.array(pred_nf.cpu()).tolist())
            pred2.extend(np.array(pred_shape.cpu()).tolist())
            true1.extend(np.array(label_nf.argmax(axis=1).cpu()).tolist())
            true2.extend(np.array(label_shape.argmax(axis=1).cpu()).tolist())
            prob.extend(np.array(prob2[:,1].cpu()).tolist())

            progress_bar.set_description(
                f'val: epoch: {epoch} cls_nf: {np.mean(loss_cls1_val):1.4f} cls_shape: {np.mean(loss_cls2_val):1.4f} loss: {np.mean(epoch_loss_val):1.4f} acc_nf: {np.mean(acc_cls1_val):1.4f} acc_shape: {np.mean(acc_cls2_val):1.4f} acc_all: {(np.mean(acc_cls1_val)+np.mean(acc_cls2_val))/2:1.4f}')
        
        # nf
        cm = confusion_matrix(np.array(true1),np.array(pred1))
        labels_name = ['胞核深染轻度,核质比增大','胞核深染轻度,核质比很大','胞核深染显著,核质比增大','胞核深染显著,核质比很大']
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        plot_confusion_matrix(cm,labels_name)
        #print('save img in temp/results/images/cm_nf.png')
        #plt.savefig('temp/results/images/cm_nf.png')
        #plt.show()

        # shape
        plot_roc(np.array(true2),np.array(prob),title='ROC(shape classification)')
        #print('save img in temp/results/images/roc_shape.png')
        #plt.savefig('temp/results/images/roc_shape.png')
        pr,rc,spec,acc = compute_pr_re(np.array(true2),np.array(pred2))
        print(f'precision: {pr:.4f}, recall: {rc:.4f}, specifity: {spec:.4f}, accuracy: {acc:.4f}')


        # loss and acc change curve in metriclog, no particular code

    if args.mission == 'grade':
        slist = np.load('fin.npy')
        with open('sample_cellfeatures1.json','r') as f:
            cf = json.load(f)
        with open('cellfeatures.json','r') as f:
            cf_all = json.load(f)
        slidelist = []
        i = 0
        for slide in slist:
            if slide not in cf and slide in cf_all and i < 10:
                slidelist.append(slide)
                i = i + 1
        
        model = create_model(args.backbone,pretrained = True,num_classes=3)
        model.load_state_dict(torch.load('temp/checkpoints/resnet34/resnet34_49.pth.tar'))

        model = model.cuda()

        #model = torch.nn.DataParallel(model).cuda()

        #retinanet = torch.nn.DataParallel(retinanet).cuda()
        testdataset = GradeImageDataset(slide_list = slidelist,
                                    img_size = 512,
                                    level = 3,
                                    is_training = True)
        testdataloader = DataLoader(testdataset,
                                    batch_size = 32,
                                    shuffle = True,
                                    drop_last = True)
        loss_fn = CrossEntropyLoss().cuda()
        #loss_fn = BCELoss().cuda()

        model.eval()
        #model.module.freeze_bn()

        print('Num training images: {}'.format(len(traindataset)))        
        with torch.no_grad():
            model.eval()
            epoch_loss_val = []
            epoch_acc_val = []
            predlist = []
            truelist = []

            progress_bar = tqdm(valdataloader)
            for imgs,label in progress_bar:
                imgs = imgs.float().cuda()
                label = label.float().cuda()

                output = model(imgs)

                output = torch.squeeze(output)
                loss = loss_fn(output,label.argmax(axis=1))
                #loss = loss_fn(output,label)

                prob = output.sigmoid()
                pred = output.argmax(axis=1)

                acc_data = (pred == label.argmax(axis=1)).sum().data * 1.0/32
                loss_data = loss.data

                epoch_loss_val.append(float(loss_data))
                epoch_acc_val.append(float(acc_data))
                predlist.extend(np.array(pred.cpu()).tolist())
                truelist.extend(np.array(label.argmax(axis=1).cpu()).tolist())

                progress_bar.set_description(
                    f'val:  epoch: {epoch} loss: {np.mean(epoch_loss_val):1.4f} acc_all: {np.mean(epoch_acc_val):1.4f}')
                        
            cm = confusion_matrix(np.array(truelist),np.array(predlist))
            labels_name = ['II级','III级','IV级']
            plt.rcParams['font.sans-serif']=['SimHei']
            plt.rcParams['axes.unicode_minus']=False
            plot_confusion_matrix(cm,labels_name)
            #print('save img in temp/results/images/cm_grade.png')
            #plt.savefig('temp/results/images/cm_grade.png')
            #plt.show()
    if args.mission == 'wsi':
        slist = np.load('fin.npy')
        with open('cftestnf.json','r') as f:
            cf = json.load(f)
        slidelist2 = []
        for slide in slist:
            if slide in cf:
                slidelist2.append(slide)
        
        model = resnet_fpn.resnet34_fpn_classifier(num_classes=[2,2,2],pretrained = True)
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load('temp/checkpoints/bestmodel/resnet34_fpn_classifier_14_3.pth.tar'))
        loss_fn = CrossEntropyLoss().cuda()
        #loss_fn = BCELoss().cuda()

        model.eval()

        preds_nf = np.zeros(len(slidelist2))
        preds_ncr = np.zeros(len(slidelist2))
        preds_shape = np.zeros(len(slidelist2))

        for i in range(len(slidelist2)):
            slide = slidelist2[i]
            print(slide)
            testdataset = WSIimageDataset(slide,1)
            testdataloader = DataLoader(testdataset,batch_size = 32,drop_last = True)

            labels = cf[slide]
            preds1 = []
            preds2 = []
            preds3 = []


            progress_bar = tqdm(enumerate(testdataloader),total = len(testdataloader))
            for iter_num,data in progress_bar:
                img = data
                img = img.cuda().float()
                #label_nf = label_nf.cuda().float()
                #label_shape = label_shape.cuda().float()
                output1,output2,output3 = model(img)

                output1 = torch.squeeze(output1)
                output2 = torch.squeeze(output2)
                output3 = torch.squeeze(output3)


                pred_nf = output1.argmax(axis=1)
                pred_ncr = output2.argmax(axis=1)
                pred_shape = output3.argmax(axis=1)

                #prob2 = output2.sigmoid()

                #print(pred_shape)
                #print(label_shape.argmax(axis=1))

                preds1.extend(np.array(pred_nf.cpu()).tolist())
                preds2.extend(np.array(pred_ncr.cpu()).tolist())
                preds3.extend(np.array(pred_shape.cpu()).tolist())


                progress_bar.set_description(f'test {slide}')

            rate1 = np.sum(preds1)/len(preds1)
            rate2 = np.sum(preds2)/len(preds2)
            rate3 = np.sum(preds3)/len(preds3)
            #print(rate1,rate2,rate3)
            if rate1 > 0.5:
                preds_nf[i] = 1
            if rate2 > 0.5:
                preds_ncr[i] = 1
            if rate3 > 0.5:
                preds_shape[i] = 1
        labels_nfname = ['轻度','显著']
        labels_ncrname = ['增大','很大']
        labels_shapename = ['粗梁实性','粗梁实性和假腺状']

        preds_ncr = preds_ncr.astype(np.int)
        preds_nf = preds_nf.astype(np.int)
        preds_shape = preds_shape.astype(np.int)


        print('==> generate pathological description')
        for i in range(len(slidelist2)):
            print(f'slide id: {slidelist2[i]}')
            print(f'该切片细胞核深染{labels_nfname[preds_nf[i]]},核质比{labels_ncrname[preds_ncr[i]]},肿瘤区域细胞主要呈{labels_shapename[preds_shape[i]]}排列。')


######## 其他分类

                




def main():
    args = parser.parse_args()
    test(args)


if __name__ == '__main__':
    main()