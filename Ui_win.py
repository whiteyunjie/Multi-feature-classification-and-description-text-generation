# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'e:\大四上\硕\毕设\segref\win.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(987, 583)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setAutoFillBackground(True)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(30, 20, 91, 31))
        self.label.setStyleSheet("font: 12pt \"微软雅黑\";\n"
"")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(350, 20, 91, 31))
        self.label_2.setStyleSheet("font: 12pt \"微软雅黑\";\n"
"")
        self.label_2.setObjectName("label_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(710, 60, 211, 261))
        self.textBrowser.setStyleSheet("font: 12pt \"楷体\";\n"
"background-color: rgb(255, 255, 255);")
        self.textBrowser.setObjectName("textBrowser")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(713, 31, 101, 21))
        self.label_4.setStyleSheet("font: 12pt \"微软雅黑\";\n"
"")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(20, 60, 321, 421))
        self.label_5.setStyleSheet("border-color: rgb(170, 255, 255);\n"
"background-color: rgb(170, 255, 255);")
        self.label_5.setFrameShape(QtWidgets.QFrame.Box)
        self.label_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_5.setLineWidth(3)
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(350, 60, 321, 421))
        self.label_6.setStyleSheet("border-color: rgb(170, 255, 255);\n"
"background-color: rgb(170, 255, 255);")
        self.label_6.setFrameShape(QtWidgets.QFrame.Box)
        self.label_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_6.setLineWidth(3)
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(710, 340, 211, 141))
        self.groupBox.setStyleSheet("")
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.pushButton = QtWidgets.QPushButton(self.groupBox)
        self.pushButton.setGeometry(QtCore.QRect(10, 90, 201, 31))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(243, 244, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(243, 244, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(243, 244, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(243, 244, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(243, 244, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(243, 244, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(243, 244, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(243, 244, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(243, 244, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.pushButton.setPalette(palette)
        self.pushButton.setStyleSheet("font: 12pt \"微软雅黑\";\n"
"\n"
"background-color: rgb(243, 244, 255);")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("run.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon1)
        self.pushButton.setObjectName("pushButton")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(10, 50, 81, 31))
        self.label_3.setStyleSheet("\n"
"font: 12pt \"微软雅黑\";\n"
"")
        self.label_3.setObjectName("label_3")
        self.comboBox = QtWidgets.QComboBox(self.groupBox)
        self.comboBox.setGeometry(QtCore.QRect(90, 50, 151, 31))
        self.comboBox.setStyleSheet("font: 12pt \"楷体\";\n"
"")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(0, 0, 981, 531))
        self.listView.setStyleSheet("background-image: url(:/imgs/b1.jpg);\n"
"")
        self.listView.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.listView.setResizeMode(QtWidgets.QListView.Adjust)
        self.listView.setObjectName("listView")
        self.listView.raise_()
        self.textBrowser.raise_()
        self.label_6.raise_()
        self.label_5.raise_()
        self.label.raise_()
        self.label_2.raise_()
        self.label_4.raise_()
        self.groupBox.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 987, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "病理切片文本描述"))
        self.label.setText(_translate("MainWindow", "原始切片"))
        self.label_2.setText(_translate("MainWindow", "肿瘤区域"))
        self.label_4.setText(_translate("MainWindow", "文本描述"))
        self.pushButton.setText(_translate("MainWindow", "生成文本描述"))
        self.label_3.setText(_translate("MainWindow", "选择切片"))
        self.comboBox.setItemText(0, _translate("MainWindow", "201601986-2"))
        self.comboBox.setItemText(1, _translate("MainWindow", "201610196-3"))
        self.comboBox.setItemText(2, _translate("MainWindow", "201434013-4"))
        self.comboBox.setItemText(3, _translate("MainWindow", "201610196-5"))
        self.comboBox.setItemText(4, _translate("MainWindow", "201439600-1"))
        self.comboBox.setItemText(5, _translate("MainWindow", "201605479-1"))
        self.comboBox.setItemText(6, _translate("MainWindow", "201436453-3"))
        self.comboBox.setItemText(7, _translate("MainWindow", "201437427-4"))
        self.comboBox.setItemText(8, _translate("MainWindow", "201609207-4"))
        self.comboBox.setItemText(9, _translate("MainWindow", "201604335-1"))
        self.comboBox.setItemText(10, _translate("MainWindow", "201613577-2"))
        self.comboBox.setItemText(11, _translate("MainWindow", "201437429-3"))
        self.comboBox.setItemText(12, _translate("MainWindow", "201605479-4"))
        self.comboBox.setItemText(13, _translate("MainWindow", "201436170-3"))
        self.comboBox.setItemText(14, _translate("MainWindow", "201434822-1"))
        self.comboBox.setItemText(15, _translate("MainWindow", "201600623-1"))
        self.comboBox.setItemText(16, _translate("MainWindow", "17-043414-1"))
        self.comboBox.setItemText(17, _translate("MainWindow", "17-043414-2"))
        self.comboBox.setItemText(18, _translate("MainWindow", "17-043414-3"))
        self.comboBox.setItemText(19, _translate("MainWindow", "17-043414-4"))
        self.comboBox.setItemText(20, _translate("MainWindow", "17-046476-1"))
        self.comboBox.setItemText(21, _translate("MainWindow", "17-046476-2"))
        self.comboBox.setItemText(22, _translate("MainWindow", "17-046476-3"))
        self.comboBox.setItemText(23, _translate("MainWindow", "17-046476-4"))
        self.comboBox.setItemText(24, _translate("MainWindow", "201629755-2"))
        self.comboBox.setItemText(25, _translate("MainWindow", "201629755-3"))
        self.comboBox.setItemText(26, _translate("MainWindow", "201629755-4"))
        self.comboBox.setItemText(27, _translate("MainWindow", "201630452-2"))
        self.comboBox.setItemText(28, _translate("MainWindow", "201630452-3"))
        self.comboBox.setItemText(29, _translate("MainWindow", "201630452-4"))
import image_rc