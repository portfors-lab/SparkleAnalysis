# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:/Users/Joel/Documents/AnalysisGui/ui/tuning_curves.ui'
#
# Created: Tue Mar 01 15:21:16 2016
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

class Ui_Form_tuning_curves(object):
    def setupUi(self, Form_tuning_curves):
        Form_tuning_curves.setObjectName(_fromUtf8("Form_tuning_curves"))
        Form_tuning_curves.setEnabled(True)
        Form_tuning_curves.resize(680, 707)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form_tuning_curves.sizePolicy().hasHeightForWidth())
        Form_tuning_curves.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("horsey.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Form_tuning_curves.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(Form_tuning_curves)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setContentsMargins(-1, -1, -1, 0)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.gridLayout_4 = QtGui.QGridLayout()
        self.gridLayout_4.setContentsMargins(-1, -1, -1, 0)
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.groupBox_comments = QtGui.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_comments.setFont(font)
        self.groupBox_comments.setFlat(True)
        self.groupBox_comments.setObjectName(_fromUtf8("groupBox_comments"))
        self.gridLayout_13 = QtGui.QGridLayout(self.groupBox_comments)
        self.gridLayout_13.setObjectName(_fromUtf8("gridLayout_13"))
        self.lineEdit_comments = QtGui.QLineEdit(self.groupBox_comments)
        self.lineEdit_comments.setEnabled(False)
        self.lineEdit_comments.setMinimumSize(QtCore.QSize(0, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.lineEdit_comments.setFont(font)
        self.lineEdit_comments.setReadOnly(True)
        self.lineEdit_comments.setObjectName(_fromUtf8("lineEdit_comments"))
        self.gridLayout_13.addWidget(self.lineEdit_comments, 0, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_comments, 0, 2, 2, 1)
        self.groupBox_test = QtGui.QGroupBox(self.centralwidget)
        self.groupBox_test.setMinimumSize(QtCore.QSize(125, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_test.setFont(font)
        self.groupBox_test.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.groupBox_test.setFlat(True)
        self.groupBox_test.setObjectName(_fromUtf8("groupBox_test"))
        self.gridLayout_10 = QtGui.QGridLayout(self.groupBox_test)
        self.gridLayout_10.setObjectName(_fromUtf8("gridLayout_10"))
        self.comboBox_test_num = QtGui.QComboBox(self.groupBox_test)
        self.comboBox_test_num.setEnabled(False)
        self.comboBox_test_num.setMinimumSize(QtCore.QSize(75, 20))
        self.comboBox_test_num.setMaximumSize(QtCore.QSize(77, 16777215))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.comboBox_test_num.setFont(font)
        self.comboBox_test_num.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.comboBox_test_num.setObjectName(_fromUtf8("comboBox_test_num"))
        self.gridLayout_10.addWidget(self.comboBox_test_num, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.gridLayout_4.addWidget(self.groupBox_test, 0, 0, 2, 1)
        self.groupBox_channel = QtGui.QGroupBox(self.centralwidget)
        self.groupBox_channel.setMinimumSize(QtCore.QSize(125, 0))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_channel.setFont(font)
        self.groupBox_channel.setFlat(True)
        self.groupBox_channel.setObjectName(_fromUtf8("groupBox_channel"))
        self.gridLayout_2 = QtGui.QGridLayout(self.groupBox_channel)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.comboBox_channel = QtGui.QComboBox(self.groupBox_channel)
        self.comboBox_channel.setEnabled(False)
        self.comboBox_channel.setMinimumSize(QtCore.QSize(75, 20))
        self.comboBox_channel.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.comboBox_channel.setFont(font)
        self.comboBox_channel.setObjectName(_fromUtf8("comboBox_channel"))
        self.gridLayout_2.addWidget(self.comboBox_channel, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.gridLayout_4.addWidget(self.groupBox_channel, 0, 1, 2, 1)
        self.verticalLayout.addLayout(self.gridLayout_4)
        self.groupBoxPlot = QtGui.QGroupBox(self.centralwidget)
        self.groupBoxPlot.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxPlot.sizePolicy().hasHeightForWidth())
        self.groupBoxPlot.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.groupBoxPlot.setFont(font)
        self.groupBoxPlot.setFlat(True)
        self.groupBoxPlot.setObjectName(_fromUtf8("groupBoxPlot"))
        self.horizontalLayout_7 = QtGui.QHBoxLayout(self.groupBoxPlot)
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.radioButtonFreq = QtGui.QRadioButton(self.groupBoxPlot)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.radioButtonFreq.setFont(font)
        self.radioButtonFreq.setChecked(True)
        self.radioButtonFreq.setObjectName(_fromUtf8("radioButtonFreq"))
        self.horizontalLayout_7.addWidget(self.radioButtonFreq)
        self.radioButtonContour = QtGui.QRadioButton(self.groupBoxPlot)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.radioButtonContour.setFont(font)
        self.radioButtonContour.setObjectName(_fromUtf8("radioButtonContour"))
        self.horizontalLayout_7.addWidget(self.radioButtonContour)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_7.addItem(spacerItem)
        self.verticalLayout.addWidget(self.groupBoxPlot)
        self.groupBoxUnits = QtGui.QGroupBox(self.centralwidget)
        self.groupBoxUnits.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBoxUnits.setFont(font)
        self.groupBoxUnits.setFlat(True)
        self.groupBoxUnits.setObjectName(_fromUtf8("groupBoxUnits"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.groupBoxUnits)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.radioButtonMeanSpikes = QtGui.QRadioButton(self.groupBoxUnits)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.radioButtonMeanSpikes.setFont(font)
        self.radioButtonMeanSpikes.setChecked(True)
        self.radioButtonMeanSpikes.setObjectName(_fromUtf8("radioButtonMeanSpikes"))
        self.horizontalLayout_3.addWidget(self.radioButtonMeanSpikes)
        self.radioButtonResponseRate = QtGui.QRadioButton(self.groupBoxUnits)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.radioButtonResponseRate.setFont(font)
        self.radioButtonResponseRate.setObjectName(_fromUtf8("radioButtonResponseRate"))
        self.horizontalLayout_3.addWidget(self.radioButtonResponseRate)
        self.radioButtonOther = QtGui.QRadioButton(self.groupBoxUnits)
        font = QtGui.QFont()
        font.setPointSize(8)
        font.setStrikeOut(True)
        self.radioButtonOther.setFont(font)
        self.radioButtonOther.setObjectName(_fromUtf8("radioButtonOther"))
        self.horizontalLayout_3.addWidget(self.radioButtonOther)
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.verticalLayout.addWidget(self.groupBoxUnits)
        self.gridLayout.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(-1, -1, -1, 9)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.label_title = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_title.setFont(font)
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title.setObjectName(_fromUtf8("label_title"))
        self.horizontalLayout.addWidget(self.label_title)
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.gridLayout_12 = QtGui.QGridLayout()
        self.gridLayout_12.setObjectName(_fromUtf8("gridLayout_12"))
        self.line_5 = QtGui.QFrame(self.centralwidget)
        self.line_5.setFrameShape(QtGui.QFrame.VLine)
        self.line_5.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_5.setObjectName(_fromUtf8("line_5"))
        self.gridLayout_12.addWidget(self.line_5, 0, 1, 1, 1)
        self.groupBoxWindow = QtGui.QGroupBox(self.centralwidget)
        self.groupBoxWindow.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBoxWindow.setFont(font)
        self.groupBoxWindow.setFlat(True)
        self.groupBoxWindow.setCheckable(True)
        self.groupBoxWindow.setChecked(False)
        self.groupBoxWindow.setObjectName(_fromUtf8("groupBoxWindow"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.groupBoxWindow)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.verticalLayout_7 = QtGui.QVBoxLayout()
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.label_xmax = QtGui.QLabel(self.groupBoxWindow)
        self.label_xmax.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_xmax.setFont(font)
        self.label_xmax.setAlignment(QtCore.Qt.AlignCenter)
        self.label_xmax.setObjectName(_fromUtf8("label_xmax"))
        self.verticalLayout_7.addWidget(self.label_xmax)
        self.doubleSpinBox_xmax = QtGui.QDoubleSpinBox(self.groupBoxWindow)
        self.doubleSpinBox_xmax.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.doubleSpinBox_xmax.setFont(font)
        self.doubleSpinBox_xmax.setDecimals(4)
        self.doubleSpinBox_xmax.setMinimum(-1.0)
        self.doubleSpinBox_xmax.setMaximum(100.0)
        self.doubleSpinBox_xmax.setSingleStep(0.001)
        self.doubleSpinBox_xmax.setProperty("value", 0.2)
        self.doubleSpinBox_xmax.setObjectName(_fromUtf8("doubleSpinBox_xmax"))
        self.verticalLayout_7.addWidget(self.doubleSpinBox_xmax)
        self.doubleSpinBox_xmin = QtGui.QDoubleSpinBox(self.groupBoxWindow)
        self.doubleSpinBox_xmin.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.doubleSpinBox_xmin.setFont(font)
        self.doubleSpinBox_xmin.setDecimals(4)
        self.doubleSpinBox_xmin.setMinimum(-1.0)
        self.doubleSpinBox_xmin.setMaximum(100.0)
        self.doubleSpinBox_xmin.setSingleStep(0.001)
        self.doubleSpinBox_xmin.setProperty("value", 0.0)
        self.doubleSpinBox_xmin.setObjectName(_fromUtf8("doubleSpinBox_xmin"))
        self.verticalLayout_7.addWidget(self.doubleSpinBox_xmin)
        self.label_xmin = QtGui.QLabel(self.groupBoxWindow)
        self.label_xmin.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_xmin.setFont(font)
        self.label_xmin.setAlignment(QtCore.Qt.AlignCenter)
        self.label_xmin.setObjectName(_fromUtf8("label_xmin"))
        self.verticalLayout_7.addWidget(self.label_xmin)
        self.horizontalLayout_4.addLayout(self.verticalLayout_7)
        self.verticalLayout_6 = QtGui.QVBoxLayout()
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.label_ymax = QtGui.QLabel(self.groupBoxWindow)
        self.label_ymax.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_ymax.setFont(font)
        self.label_ymax.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ymax.setObjectName(_fromUtf8("label_ymax"))
        self.verticalLayout_6.addWidget(self.label_ymax)
        self.doubleSpinBox_ymax = QtGui.QDoubleSpinBox(self.groupBoxWindow)
        self.doubleSpinBox_ymax.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.doubleSpinBox_ymax.setFont(font)
        self.doubleSpinBox_ymax.setDecimals(4)
        self.doubleSpinBox_ymax.setMinimum(-100.0)
        self.doubleSpinBox_ymax.setMaximum(100.0)
        self.doubleSpinBox_ymax.setSingleStep(0.001)
        self.doubleSpinBox_ymax.setProperty("value", 0.1)
        self.doubleSpinBox_ymax.setObjectName(_fromUtf8("doubleSpinBox_ymax"))
        self.verticalLayout_6.addWidget(self.doubleSpinBox_ymax)
        self.doubleSpinBox_ymin = QtGui.QDoubleSpinBox(self.groupBoxWindow)
        self.doubleSpinBox_ymin.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.doubleSpinBox_ymin.setFont(font)
        self.doubleSpinBox_ymin.setDecimals(4)
        self.doubleSpinBox_ymin.setMinimum(-100.0)
        self.doubleSpinBox_ymin.setMaximum(100.0)
        self.doubleSpinBox_ymin.setSingleStep(0.001)
        self.doubleSpinBox_ymin.setProperty("value", -0.1)
        self.doubleSpinBox_ymin.setObjectName(_fromUtf8("doubleSpinBox_ymin"))
        self.verticalLayout_6.addWidget(self.doubleSpinBox_ymin)
        self.label_ymin = QtGui.QLabel(self.groupBoxWindow)
        self.label_ymin.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_ymin.setFont(font)
        self.label_ymin.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ymin.setObjectName(_fromUtf8("label_ymin"))
        self.verticalLayout_6.addWidget(self.label_ymin)
        self.horizontalLayout_4.addLayout(self.verticalLayout_6)
        self.gridLayout_12.addWidget(self.groupBoxWindow, 0, 0, 1, 1)
        self.pushButtonGenerate = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButtonGenerate.sizePolicy().hasHeightForWidth())
        self.pushButtonGenerate.setSizePolicy(sizePolicy)
        self.pushButtonGenerate.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.pushButtonGenerate.setAutoDefault(False)
        self.pushButtonGenerate.setObjectName(_fromUtf8("pushButtonGenerate"))
        self.gridLayout_12.addWidget(self.pushButtonGenerate, 1, 2, 1, 1)
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.groupBoxThreshold = QtGui.QGroupBox(self.centralwidget)
        self.groupBoxThreshold.setMinimumSize(QtCore.QSize(0, 100))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBoxThreshold.setFont(font)
        self.groupBoxThreshold.setFlat(True)
        self.groupBoxThreshold.setObjectName(_fromUtf8("groupBoxThreshold"))
        self.gridLayout_6 = QtGui.QGridLayout(self.groupBoxThreshold)
        self.gridLayout_6.setObjectName(_fromUtf8("gridLayout_6"))
        self.doubleSpinBox_threshold = QtGui.QDoubleSpinBox(self.groupBoxThreshold)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.doubleSpinBox_threshold.setFont(font)
        self.doubleSpinBox_threshold.setDecimals(4)
        self.doubleSpinBox_threshold.setMinimum(-100.0)
        self.doubleSpinBox_threshold.setMaximum(100.0)
        self.doubleSpinBox_threshold.setSingleStep(0.001)
        self.doubleSpinBox_threshold.setObjectName(_fromUtf8("doubleSpinBox_threshold"))
        self.gridLayout_6.addWidget(self.doubleSpinBox_threshold, 1, 0, 1, 1)
        self.pushButton_auto_threshold = QtGui.QPushButton(self.groupBoxThreshold)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.pushButton_auto_threshold.setFont(font)
        self.pushButton_auto_threshold.setObjectName(_fromUtf8("pushButton_auto_threshold"))
        self.gridLayout_6.addWidget(self.pushButton_auto_threshold, 1, 1, 1, 1)
        self.gridLayout_3.addWidget(self.groupBoxThreshold, 2, 0, 1, 1)
        self.line = QtGui.QFrame(self.centralwidget)
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.gridLayout_3.addWidget(self.line, 0, 1, 4, 1)
        self.groupBox_stimulus = QtGui.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_stimulus.setFont(font)
        self.groupBox_stimulus.setFlat(True)
        self.groupBox_stimulus.setObjectName(_fromUtf8("groupBox_stimulus"))
        self.gridLayout_8 = QtGui.QGridLayout(self.groupBox_stimulus)
        self.gridLayout_8.setObjectName(_fromUtf8("gridLayout_8"))
        self.label_stim_type = QtGui.QLabel(self.groupBox_stimulus)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_stim_type.setFont(font)
        self.label_stim_type.setObjectName(_fromUtf8("label_stim_type"))
        self.gridLayout_8.addWidget(self.label_stim_type, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox_stimulus, 0, 0, 2, 1)
        self.groupBox_log = QtGui.QGroupBox(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_log.sizePolicy().hasHeightForWidth())
        self.groupBox_log.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_log.setFont(font)
        self.groupBox_log.setFlat(True)
        self.groupBox_log.setObjectName(_fromUtf8("groupBox_log"))
        self.gridLayout_9 = QtGui.QGridLayout(self.groupBox_log)
        self.gridLayout_9.setObjectName(_fromUtf8("gridLayout_9"))
        self.textEdit = QtGui.QTextEdit(self.groupBox_log)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.textEdit.setFont(font)
        self.textEdit.setObjectName(_fromUtf8("textEdit"))
        self.gridLayout_9.addWidget(self.textEdit, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox_log, 0, 2, 3, 1)
        self.gridLayout_12.addLayout(self.gridLayout_3, 0, 2, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_12, 3, 0, 1, 1)
        self.gridLayout_14 = QtGui.QGridLayout()
        self.gridLayout_14.setObjectName(_fromUtf8("gridLayout_14"))
        self.groupBox_view = QtGui.QGroupBox(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_view.sizePolicy().hasHeightForWidth())
        self.groupBox_view.setSizePolicy(sizePolicy)
        self.groupBox_view.setMinimumSize(QtCore.QSize(0, 200))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_view.setFont(font)
        self.groupBox_view.setFlat(True)
        self.groupBox_view.setObjectName(_fromUtf8("groupBox_view"))
        self.gridLayout_11 = QtGui.QGridLayout(self.groupBox_view)
        self.gridLayout_11.setObjectName(_fromUtf8("gridLayout_11"))
        self.view = TraceWidget(self.groupBox_view)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.view.sizePolicy().hasHeightForWidth())
        self.view.setSizePolicy(sizePolicy)
        self.view.setMinimumSize(QtCore.QSize(0, 0))
        self.view.setObjectName(_fromUtf8("view"))
        self.gridLayout_11.addWidget(self.view, 1, 0, 1, 1)
        self.comboBox_trace = QtGui.QComboBox(self.groupBox_view)
        self.comboBox_trace.setEnabled(False)
        self.comboBox_trace.setMinimumSize(QtCore.QSize(75, 0))
        self.comboBox_trace.setMaximumSize(QtCore.QSize(77, 16777215))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.comboBox_trace.setFont(font)
        self.comboBox_trace.setObjectName(_fromUtf8("comboBox_trace"))
        self.gridLayout_11.addWidget(self.comboBox_trace, 0, 0, 1, 1)
        self.gridLayout_14.addWidget(self.groupBox_view, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_14, 2, 0, 1, 1)
        Form_tuning_curves.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(Form_tuning_curves)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 680, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        Form_tuning_curves.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(Form_tuning_curves)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        Form_tuning_curves.setStatusBar(self.statusbar)

        self.retranslateUi(Form_tuning_curves)
        QtCore.QMetaObject.connectSlotsByName(Form_tuning_curves)

    def retranslateUi(self, Form_tuning_curves):
        Form_tuning_curves.setWindowTitle(_translate("Form_tuning_curves", "Sparkle Analysis", None))
        self.groupBox_comments.setTitle(_translate("Form_tuning_curves", "Comments ", None))
        self.groupBox_test.setTitle(_translate("Form_tuning_curves", "Test Number ", None))
        self.groupBox_channel.setTitle(_translate("Form_tuning_curves", "Channel Number ", None))
        self.groupBoxPlot.setTitle(_translate("Form_tuning_curves", "Plot Type ", None))
        self.radioButtonFreq.setText(_translate("Form_tuning_curves", "Frequency Response", None))
        self.radioButtonContour.setText(_translate("Form_tuning_curves", "Contour Plot", None))
        self.groupBoxUnits.setTitle(_translate("Form_tuning_curves", "Unit Type ", None))
        self.radioButtonMeanSpikes.setText(_translate("Form_tuning_curves", "Mean Spikes Per Presentation", None))
        self.radioButtonResponseRate.setText(_translate("Form_tuning_curves", "Response Rate (Hz)", None))
        self.radioButtonOther.setText(_translate("Form_tuning_curves", "Other", None))
        self.label_title.setText(_translate("Form_tuning_curves", "Test Name", None))
        self.groupBoxWindow.setTitle(_translate("Form_tuning_curves", "Use Custom Window ", None))
        self.label_xmax.setText(_translate("Form_tuning_curves", "X Max", None))
        self.doubleSpinBox_xmax.setSuffix(_translate("Form_tuning_curves", " s", None))
        self.doubleSpinBox_xmin.setSuffix(_translate("Form_tuning_curves", " s", None))
        self.label_xmin.setText(_translate("Form_tuning_curves", "X Min", None))
        self.label_ymax.setText(_translate("Form_tuning_curves", "Y Max", None))
        self.doubleSpinBox_ymax.setSuffix(_translate("Form_tuning_curves", " V", None))
        self.doubleSpinBox_ymin.setSuffix(_translate("Form_tuning_curves", " V", None))
        self.label_ymin.setText(_translate("Form_tuning_curves", "Y Min", None))
        self.pushButtonGenerate.setText(_translate("Form_tuning_curves", "Generate Plot", None))
        self.groupBoxThreshold.setTitle(_translate("Form_tuning_curves", "Threshold ", None))
        self.doubleSpinBox_threshold.setSuffix(_translate("Form_tuning_curves", " V", None))
        self.pushButton_auto_threshold.setText(_translate("Form_tuning_curves", "Estimate Threshold", None))
        self.groupBox_stimulus.setTitle(_translate("Form_tuning_curves", "Stimulus Type ", None))
        self.label_stim_type.setText(_translate("Form_tuning_curves", "None", None))
        self.groupBox_log.setTitle(_translate("Form_tuning_curves", "Log ", None))
        self.groupBox_view.setTitle(_translate("Form_tuning_curves", "Trace View ", None))

from util.pyqtgraph_widgets import TraceWidget
