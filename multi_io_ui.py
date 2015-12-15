# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:/Users/Joel/Documents/AnalysisGui/multi_io.ui'
#
# Created: Mon Dec 14 15:35:55 2015
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Form_multi_io(object):
    def setupUi(self, Form_multi_io):
        Form_multi_io.setObjectName(_fromUtf8("Form_multi_io"))
        Form_multi_io.setEnabled(True)
        Form_multi_io.resize(300, 400)
        Form_multi_io.setAcceptDrops(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("horsey.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form_multi_io.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(Form_multi_io)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.pushButton_multi_io = QtGui.QPushButton(self.centralwidget)
        self.pushButton_multi_io.setObjectName(_fromUtf8("pushButton_multi_io"))
        self.gridLayout_2.addWidget(self.pushButton_multi_io, 0, 2, 1, 1)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 0, 0, 1, 1)
        self.pushButton_auto_threshold = QtGui.QPushButton(self.centralwidget)
        self.pushButton_auto_threshold.setObjectName(_fromUtf8("pushButton_auto_threshold"))
        self.gridLayout_2.addWidget(self.pushButton_auto_threshold, 0, 1, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 6, 0, 1, 1)
        self.scrollArea = QtGui.QScrollArea(self.centralwidget)
        self.scrollArea.setFrameShape(QtGui.QFrame.Panel)
        self.scrollArea.setFrameShadow(QtGui.QFrame.Sunken)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 280, 280))
        self.scrollAreaWidgetContents.setObjectName(_fromUtf8("scrollAreaWidgetContents"))
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 5, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.label_title = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.horizontalLayout.addWidget(self.label_title)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        Form_multi_io.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(Form_multi_io)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 300, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        Form_multi_io.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(Form_multi_io)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        Form_multi_io.setStatusBar(self.statusbar)

        self.retranslateUi(Form_multi_io)
        QtCore.QMetaObject.connectSlotsByName(Form_multi_io)

    def retranslateUi(self, Form_multi_io):
        Form_multi_io.setWindowTitle(_translate("Form_multi_io", "Multi I/O Plot", None))
        self.pushButton_multi_io.setToolTip(_translate("Form_multi_io", "Graphs I/O tests based on the selected tests and their thresholds.", None))
        self.pushButton_multi_io.setText(_translate("Form_multi_io", "Plot", None))
        self.pushButton_auto_threshold.setToolTip(_translate("Form_multi_io", "Estimates the thresholds of all the tests.", None))
        self.pushButton_auto_threshold.setText(_translate("Form_multi_io", "Estimate Thresholds", None))
        self.label_title.setText(_translate("Form_multi_io", "Test Name", None))

