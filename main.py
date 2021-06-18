import Ui_win
import Ui_list
import numpy as np
import json
import sys
import os
import cv2
from PIL import Image

from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class mainwindow(QMainWindow,Ui_win.Ui_MainWindow):
    #overprogram = QtCore.pyqtSignal()
    def __init__(self,parent=None,fri={},hostid=''):
        super(mainwindow,self).__init__(parent)
        self.setupUi(self)
        #bg = QPalette()
        #bg.setBrush(self.backgroundRole(),QBrush(QPixmap('b1.jpg')))
        #self.setPalette(bg)
        #self.set_item(fri) 

class mainwindowlist(QMainWindow,Ui_list.Ui_MainWindow):
    overprogram = QtCore.pyqtSignal()
    def __init__(self,preds,parent=None):
        super(mainwindowlist,self).__init__(parent)
        self.setupUi(self)
        self.preds = preds
        self.set_item()

    def set_item(self):
        self.listWidget.clear()
        for slidename in self.preds:
            layoutmain = QHBoxLayout()
            slideform = QLabel()
            layoutmain.addWidget(slideform)


            map = QPixmap('slideimgs/'+slidename+'.jpg').scaled(150,200)
            imag = QLabel()  #头像显示，暂不用
            imag.setPixmap(map)
            imag.resize(200,200)
            #imag.setScaledContents(True)
            layoutmain.addWidget(imag)
            #layoutmain.addWidget(QLabel(''))
            labels_nfname = ['轻度','显著']
            labels_ncrname = ['增大','很大']
            labels_shapename = ['粗梁实性','粗梁实性和假腺状']
            labels_name = ['无','有']
            labels_gradename = ['II','III','IV']
            
            des = '该切片有' + labels_nfname[self.preds[slidename][1]] +\
                '细胞核深染现象，核质比'+ labels_ncrname[self.preds[slidename][2]]+\
                '，肿瘤区域细胞主要呈' + labels_shapename[self.preds[slidename][3]]+\
                '排列，'+ labels_name[self.preds[slidename][4]] + '明显的细胞坏死现象，' +\
                labels_name[self.preds[slidename][5]] + '包膜，分级为' + labels_gradename[self.preds[slidename][0]]+'级'

            desform = QLabel(des)
            desform.setStyleSheet("font: 12pt \"楷体\";\n""")
            desform.setWordWrap(True)
            desform.resize(300,100)
            layoutmain.addWidget(desform)

            item = QListWidgetItem()
            item.setSizeHint(QtCore.QSize(500,200))
            item.setText(slidename)
            widget = QWidget()
            widget.setLayout(layoutmain)
            self.listWidget.addItem(item)
            self.listWidget.setItemWidget(item,widget)
        #self.listWidget.setWordWrap(True)
        


class mainoperation():
    def __init__(self):
        self.win = mainwindow()
        self.win.pushButton.clicked.connect(self.display)
        self.slidename = self.win.comboBox.currentText()
        with open('preds.json','r') as f:
            self.preds = json.load(f)
        self.listwin = mainwindowlist(self.preds)
        self.listwin.listWidget.itemDoubleClicked.connect(self.display2)

    def display2(self):
        item = self.listwin.listWidget.currentItem()
        self.slidename = item.data(0)
        self.win.show()

        #print(self.slidename)
        self.win.comboBox.setCurrentText(self.slidename)
        pixmap1 = QPixmap('slideimgs/'+self.slidename+'.jpg')
        imgo = Image.open('slideimgs/'+self.slidename+'.jpg')
        mask = np.load('mask/'+self.slidename+'_mask.npy')
        maskpro = mask.astype(np.uint8)
        maskpro = np.array([maskpro for i in range(3)]).transpose(1,2,0)
        maskpro = cv2.resize(maskpro,dsize=(int(mask.shape[1]/2),int(mask.shape[0]/2)))
        X,Y = np.where(maskpro[:,:,0])
        maskpro = maskpro.transpose(1,0,2)
        img = Image.fromarray(maskpro*imgo)
        img = img.toqpixmap()
        pixmap2 = QPixmap(img)
        
        labels_nfname = ['轻度','显著']
        labels_ncrname = ['增大','很大']
        labels_shapename = ['粗梁实性','粗梁实性和假腺状']
        labels_name = ['无','有']
        labels_gradename = ['II','III','IV']
        
        des = '该切片有' + labels_nfname[self.preds[self.slidename][1]] +\
            '细胞核深染现象，核质比'+ labels_ncrname[self.preds[self.slidename][2]]+\
            '，肿瘤区域细胞主要呈' + labels_shapename[self.preds[self.slidename][3]]+\
            '排列，'+ labels_name[self.preds[self.slidename][4]] + '明显的细胞坏死现象，' +\
            labels_name[self.preds[self.slidename][5]] + '包膜，分级为' + labels_gradename[self.preds[self.slidename][0]]+'级'

        self.win.label_5.setPixmap(pixmap1)
        self.win.label_5.setScaledContents(True)
        

        self.win.label_6.setPixmap(pixmap2)
        self.win.label_6.setScaledContents(True)

        self.win.textBrowser.setText(des)

    def display(self):
        self.slidename = self.win.comboBox.currentText()
        pixmap1 = QPixmap('slideimgs/'+self.slidename+'.jpg')
        imgo = Image.open('slideimgs/'+self.slidename+'.jpg')
        mask = np.load('mask/'+self.slidename+'_mask.npy')
        maskpro = mask.astype(np.uint8)
        maskpro = np.array([maskpro for i in range(3)]).transpose(1,2,0)
        maskpro = cv2.resize(maskpro,dsize=(int(mask.shape[1]/2),int(mask.shape[0]/2)))
        X,Y = np.where(maskpro[:,:,0])
        maskpro = maskpro.transpose(1,0,2)
        img = Image.fromarray(maskpro*imgo)
        img = img.toqpixmap()
        pixmap2 = QPixmap(img)
        
        labels_nfname = ['轻度','显著']
        labels_ncrname = ['增大','很大']
        labels_shapename = ['粗梁实性','粗梁实性和假腺状']
        #des = f'该切片细胞核深染{labels_nfname[self.preds[self.slidename][0]]},核质比{labels_ncrname[self.preds[self.slidename][1]]},肿瘤区域细胞主要呈{labels_shapename[self.preds[self.slidename][3]]}排列。'
        des = '该切片有轻度细胞核深染现象,核质比较大,肿瘤区域细胞主要呈粗梁实性和假腺状排列，有明显的细胞坏死现象，有包膜，分级为II级'

        self.win.label_5.setPixmap(pixmap1)
        self.win.label_5.setScaledContents(True)
        

        self.win.label_6.setPixmap(pixmap2)
        self.win.label_6.setScaledContents(True)

        self.win.textBrowser.setText(des)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    logoper = mainoperation()
    logoper.listwin.show()
    sys.exit(app.exec_())
    os.system("pause")