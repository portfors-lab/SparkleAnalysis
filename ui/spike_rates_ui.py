# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:/Users/Joel/Documents/AnalysisGui/spike_rates.ui'
#
# Created: Thu Dec 10 13:35:40 2015
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

class Ui_Form_spike_rates(object):
    def setupUi(self, Form_spike_rates):
        Form_spike_rates.setObjectName(_fromUtf8("Form_spike_rates"))
        Form_spike_rates.setEnabled(True)
        Form_spike_rates.resize(300, 400)
        Form_spike_rates.setAcceptDrops(False)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("horsey.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form_spike_rates.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(Form_spike_rates)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_min = QtGui.QLabel(self.centralwidget)
        self.label_min.setObjectName(_fromUtf8("label_min"))
        self.horizontalLayout_2.addWidget(self.label_min)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.label_max = QtGui.QLabel(self.centralwidget)
        self.label_max.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_max.setObjectName(_fromUtf8("label_max"))
        self.horizontalLayout_2.addWidget(self.label_max)
        self.gridLayout.addLayout(self.horizontalLayout_2, 6, 0, 1, 1)
        self.horizontalSlider = QtGui.QSlider(self.centralwidget)
        self.horizontalSlider.setSliderPosition(50)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName(_fromUtf8("horizontalSlider"))
        self.gridLayout.addWidget(self.horizontalSlider, 7, 0, 1, 1)
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
        self.scrollArea = QtGui.QScrollArea(self.centralwidget)
        self.scrollArea.setFrameShape(QtGui.QFrame.Panel)
        self.scrollArea.setFrameShadow(QtGui.QFrame.Sunken)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName(_fromUtf8("scrollArea"))
        self.scrollAreaWidgetContents = QtGui.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 280, 199))
        self.scrollAreaWidgetContents.setObjectName(_fromUtf8("scrollAreaWidgetContents"))
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.gridLayout.addWidget(self.scrollArea, 5, 0, 1, 1)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.pushButton_spike_rates = QtGui.QPushButton(self.centralwidget)
        self.pushButton_spike_rates.setObjectName(_fromUtf8("pushButton_spike_rates"))
        self.gridLayout_2.addWidget(self.pushButton_spike_rates, 0, 1, 1, 1)
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem3, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 9, 0, 1, 1)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        spacerItem4 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem4)
        self.doubleSpinBox = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox.setObjectName(_fromUtf8("doubleSpinBox"))
        self.horizontalLayout_3.addWidget(self.doubleSpinBox)
        spacerItem5 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem5)
        self.gridLayout.addLayout(self.horizontalLayout_3, 8, 0, 1, 1)
        Form_spike_rates.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(Form_spike_rates)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 300, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        Form_spike_rates.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(Form_spike_rates)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        Form_spike_rates.setStatusBar(self.statusbar)

        self.retranslateUi(Form_spike_rates)
        QtCore.QMetaObject.connectSlotsByName(Form_spike_rates)

    def retranslateUi(self, Form_spike_rates):
        Form_spike_rates.setWindowTitle(_translate("Form_spike_rates", "Spike Rates", None))
        self.label_min.setText(_translate("Form_spike_rates", "Min", None))
        self.label_max.setText(_translate("Form_spike_rates", "Max", None))
        self.label_title.setText(_translate("Form_spike_rates", "Test Name", None))
        self.pushButton_spike_rates.setText(_translate("Form_spike_rates", "Plot", None))

