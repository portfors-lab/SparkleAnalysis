# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:/Users/Joel/Documents/AnalysisGui/ui/main.ui'
#
# Created: Tue Mar 01 15:51:59 2016
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

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.setEnabled(True)
        MainWindow.resize(681, 650)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("horsey.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.label = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout_3.addWidget(self.label)
        self.horizontalLayout_Open = QtGui.QHBoxLayout()
        self.horizontalLayout_Open.setObjectName(_fromUtf8("horizontalLayout_Open"))
        self.lineEdit_file_name = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_file_name.setObjectName(_fromUtf8("lineEdit_file_name"))
        self.horizontalLayout_Open.addWidget(self.lineEdit_file_name)
        self.pushButton_browse = QtGui.QPushButton(self.centralwidget)
        self.pushButton_browse.setObjectName(_fromUtf8("pushButton_browse"))
        self.horizontalLayout_Open.addWidget(self.pushButton_browse)
        self.verticalLayout_3.addLayout(self.horizontalLayout_Open)
        self.gridLayout_4 = QtGui.QGridLayout()
        self.gridLayout_4.setObjectName(_fromUtf8("gridLayout_4"))
        self.label_comments = QtGui.QLabel(self.centralwidget)
        self.label_comments.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_comments.setFont(font)
        self.label_comments.setObjectName(_fromUtf8("label_comments"))
        self.gridLayout_4.addWidget(self.label_comments, 0, 3, 1, 1)
        self.label_test_num = QtGui.QLabel(self.centralwidget)
        self.label_test_num.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_test_num.setFont(font)
        self.label_test_num.setObjectName(_fromUtf8("label_test_num"))
        self.gridLayout_4.addWidget(self.label_test_num, 0, 0, 1, 1)
        self.comboBox_test_num = QtGui.QComboBox(self.centralwidget)
        self.comboBox_test_num.setEnabled(False)
        self.comboBox_test_num.setObjectName(_fromUtf8("comboBox_test_num"))
        self.gridLayout_4.addWidget(self.comboBox_test_num, 1, 0, 1, 1)
        self.lineEdit_comments = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit_comments.setEnabled(False)
        self.lineEdit_comments.setReadOnly(True)
        self.lineEdit_comments.setObjectName(_fromUtf8("lineEdit_comments"))
        self.gridLayout_4.addWidget(self.lineEdit_comments, 1, 3, 1, 1)
        spacerItem = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem, 1, 1, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_4)
        self.horizontalLayout_Test = QtGui.QHBoxLayout()
        self.horizontalLayout_Test.setObjectName(_fromUtf8("horizontalLayout_Test"))
        self.verticalLayout_3.addLayout(self.horizontalLayout_Test)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.verticalLayout_Trace = QtGui.QVBoxLayout()
        self.verticalLayout_Trace.setObjectName(_fromUtf8("verticalLayout_Trace"))
        self.label_trace = QtGui.QLabel(self.centralwidget)
        self.label_trace.setEnabled(False)
        self.label_trace.setObjectName(_fromUtf8("label_trace"))
        self.verticalLayout_Trace.addWidget(self.label_trace)
        self.horizontalLayout_14 = QtGui.QHBoxLayout()
        self.horizontalLayout_14.setObjectName(_fromUtf8("horizontalLayout_14"))
        self.comboBox_trace = QtGui.QComboBox(self.centralwidget)
        self.comboBox_trace.setEnabled(False)
        self.comboBox_trace.setObjectName(_fromUtf8("comboBox_trace"))
        self.horizontalLayout_14.addWidget(self.comboBox_trace)
        self.verticalLayout_Trace.addLayout(self.horizontalLayout_14)
        self.horizontalLayout_4.addLayout(self.verticalLayout_Trace)
        self.verticalLayout_Repetition = QtGui.QVBoxLayout()
        self.verticalLayout_Repetition.setObjectName(_fromUtf8("verticalLayout_Repetition"))
        self.label_rep = QtGui.QLabel(self.centralwidget)
        self.label_rep.setEnabled(False)
        self.label_rep.setObjectName(_fromUtf8("label_rep"))
        self.verticalLayout_Repetition.addWidget(self.label_rep)
        self.horizontalLayout_12 = QtGui.QHBoxLayout()
        self.horizontalLayout_12.setObjectName(_fromUtf8("horizontalLayout_12"))
        self.comboBox_rep = QtGui.QComboBox(self.centralwidget)
        self.comboBox_rep.setEnabled(False)
        self.comboBox_rep.setObjectName(_fromUtf8("comboBox_rep"))
        self.horizontalLayout_12.addWidget(self.comboBox_rep)
        self.verticalLayout_Repetition.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_4.addLayout(self.verticalLayout_Repetition)
        self.verticalLayout_Channel = QtGui.QVBoxLayout()
        self.verticalLayout_Channel.setObjectName(_fromUtf8("verticalLayout_Channel"))
        self.label_channel = QtGui.QLabel(self.centralwidget)
        self.label_channel.setEnabled(False)
        self.label_channel.setObjectName(_fromUtf8("label_channel"))
        self.verticalLayout_Channel.addWidget(self.label_channel)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.comboBox_channel = QtGui.QComboBox(self.centralwidget)
        self.comboBox_channel.setEnabled(False)
        self.comboBox_channel.setObjectName(_fromUtf8("comboBox_channel"))
        self.horizontalLayout_9.addWidget(self.comboBox_channel)
        self.verticalLayout_Channel.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_4.addLayout(self.verticalLayout_Channel)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.gridLayout_2 = QtGui.QGridLayout()
        self.gridLayout_2.setContentsMargins(-1, 10, -1, 10)
        self.gridLayout_2.setObjectName(_fromUtf8("gridLayout_2"))
        self.view = TraceWidget(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.view.sizePolicy().hasHeightForWidth())
        self.view.setSizePolicy(sizePolicy)
        self.view.setObjectName(_fromUtf8("view"))
        self.gridLayout_2.addWidget(self.view, 0, 0, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_2)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        self.gridLayout.addLayout(self.verticalLayout, 1, 0, 1, 1)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        spacerItem1 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.Title = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.Title.setFont(font)
        self.Title.setObjectName(_fromUtf8("Title"))
        self.horizontalLayout.addWidget(self.Title)
        spacerItem2 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.verticalLayout_5 = QtGui.QVBoxLayout()
        self.verticalLayout_5.setContentsMargins(-1, -1, 10, -1)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.checkBox_custom_window = QtGui.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.checkBox_custom_window.setFont(font)
        self.checkBox_custom_window.setObjectName(_fromUtf8("checkBox_custom_window"))
        self.verticalLayout_5.addWidget(self.checkBox_custom_window)
        self.line = QtGui.QFrame(self.centralwidget)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.verticalLayout_5.addWidget(self.line)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.verticalLayout_7 = QtGui.QVBoxLayout()
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.label_xmax = QtGui.QLabel(self.centralwidget)
        self.label_xmax.setEnabled(False)
        self.label_xmax.setAlignment(QtCore.Qt.AlignCenter)
        self.label_xmax.setObjectName(_fromUtf8("label_xmax"))
        self.verticalLayout_7.addWidget(self.label_xmax)
        self.doubleSpinBox_xmax = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_xmax.setEnabled(False)
        self.doubleSpinBox_xmax.setDecimals(4)
        self.doubleSpinBox_xmax.setMinimum(-1.0)
        self.doubleSpinBox_xmax.setMaximum(100.0)
        self.doubleSpinBox_xmax.setSingleStep(0.001)
        self.doubleSpinBox_xmax.setProperty("value", 0.2)
        self.doubleSpinBox_xmax.setObjectName(_fromUtf8("doubleSpinBox_xmax"))
        self.verticalLayout_7.addWidget(self.doubleSpinBox_xmax)
        self.doubleSpinBox_xmin = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_xmin.setEnabled(False)
        self.doubleSpinBox_xmin.setDecimals(4)
        self.doubleSpinBox_xmin.setMinimum(-1.0)
        self.doubleSpinBox_xmin.setMaximum(100.0)
        self.doubleSpinBox_xmin.setSingleStep(0.001)
        self.doubleSpinBox_xmin.setProperty("value", 0.0)
        self.doubleSpinBox_xmin.setObjectName(_fromUtf8("doubleSpinBox_xmin"))
        self.verticalLayout_7.addWidget(self.doubleSpinBox_xmin)
        self.label_xmin = QtGui.QLabel(self.centralwidget)
        self.label_xmin.setEnabled(False)
        self.label_xmin.setAlignment(QtCore.Qt.AlignCenter)
        self.label_xmin.setObjectName(_fromUtf8("label_xmin"))
        self.verticalLayout_7.addWidget(self.label_xmin)
        self.horizontalLayout_5.addLayout(self.verticalLayout_7)
        self.verticalLayout_6 = QtGui.QVBoxLayout()
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.label_ymax = QtGui.QLabel(self.centralwidget)
        self.label_ymax.setEnabled(False)
        self.label_ymax.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ymax.setObjectName(_fromUtf8("label_ymax"))
        self.verticalLayout_6.addWidget(self.label_ymax)
        self.doubleSpinBox_ymax = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_ymax.setEnabled(False)
        self.doubleSpinBox_ymax.setDecimals(4)
        self.doubleSpinBox_ymax.setMinimum(-100.0)
        self.doubleSpinBox_ymax.setMaximum(100.0)
        self.doubleSpinBox_ymax.setSingleStep(0.001)
        self.doubleSpinBox_ymax.setProperty("value", 0.1)
        self.doubleSpinBox_ymax.setObjectName(_fromUtf8("doubleSpinBox_ymax"))
        self.verticalLayout_6.addWidget(self.doubleSpinBox_ymax)
        self.doubleSpinBox_ymin = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_ymin.setEnabled(False)
        self.doubleSpinBox_ymin.setDecimals(4)
        self.doubleSpinBox_ymin.setMinimum(-100.0)
        self.doubleSpinBox_ymin.setMaximum(100.0)
        self.doubleSpinBox_ymin.setSingleStep(0.001)
        self.doubleSpinBox_ymin.setProperty("value", -0.1)
        self.doubleSpinBox_ymin.setObjectName(_fromUtf8("doubleSpinBox_ymin"))
        self.verticalLayout_6.addWidget(self.doubleSpinBox_ymin)
        self.label_ymin = QtGui.QLabel(self.centralwidget)
        self.label_ymin.setEnabled(False)
        self.label_ymin.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ymin.setObjectName(_fromUtf8("label_ymin"))
        self.verticalLayout_6.addWidget(self.label_ymin)
        self.horizontalLayout_5.addLayout(self.verticalLayout_6)
        self.verticalLayout_5.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_2.addLayout(self.verticalLayout_5)
        self.line_4 = QtGui.QFrame(self.centralwidget)
        self.line_4.setFrameShape(QtGui.QFrame.VLine)
        self.line_4.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_4.setObjectName(_fromUtf8("line_4"))
        self.horizontalLayout_2.addWidget(self.line_4)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(10, -1, 10, -1)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.label_2 = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout_2.addWidget(self.label_2)
        self.line_2 = QtGui.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtGui.QFrame.HLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.verticalLayout_2.addWidget(self.line_2)
        self.horizontalLayout_Threshold = QtGui.QHBoxLayout()
        self.horizontalLayout_Threshold.setObjectName(_fromUtf8("horizontalLayout_Threshold"))
        self.doubleSpinBox_threshold = QtGui.QDoubleSpinBox(self.centralwidget)
        self.doubleSpinBox_threshold.setDecimals(4)
        self.doubleSpinBox_threshold.setMinimum(-100.0)
        self.doubleSpinBox_threshold.setMaximum(100.0)
        self.doubleSpinBox_threshold.setSingleStep(0.001)
        self.doubleSpinBox_threshold.setObjectName(_fromUtf8("doubleSpinBox_threshold"))
        self.horizontalLayout_Threshold.addWidget(self.doubleSpinBox_threshold)
        self.pushButton_auto_threshold = QtGui.QPushButton(self.centralwidget)
        self.pushButton_auto_threshold.setObjectName(_fromUtf8("pushButton_auto_threshold"))
        self.horizontalLayout_Threshold.addWidget(self.pushButton_auto_threshold)
        self.verticalLayout_2.addLayout(self.horizontalLayout_Threshold)
        self.label_5 = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.verticalLayout_2.addWidget(self.label_5)
        self.line_3 = QtGui.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtGui.QFrame.HLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.verticalLayout_2.addWidget(self.line_3)
        self.horizontalLayout_Polarity = QtGui.QHBoxLayout()
        self.horizontalLayout_Polarity.setObjectName(_fromUtf8("horizontalLayout_Polarity"))
        self.radioButton_normal = QtGui.QRadioButton(self.centralwidget)
        self.radioButton_normal.setChecked(True)
        self.radioButton_normal.setObjectName(_fromUtf8("radioButton_normal"))
        self.horizontalLayout_Polarity.addWidget(self.radioButton_normal)
        self.radioButton_inverse = QtGui.QRadioButton(self.centralwidget)
        self.radioButton_inverse.setObjectName(_fromUtf8("radioButton_inverse"))
        self.horizontalLayout_Polarity.addWidget(self.radioButton_inverse)
        self.verticalLayout_2.addLayout(self.horizontalLayout_Polarity)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.line_5 = QtGui.QFrame(self.centralwidget)
        self.line_5.setFrameShape(QtGui.QFrame.VLine)
        self.line_5.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_5.setObjectName(_fromUtf8("line_5"))
        self.horizontalLayout_2.addWidget(self.line_5)
        self.gridLayout_5 = QtGui.QGridLayout()
        self.gridLayout_5.setObjectName(_fromUtf8("gridLayout_5"))
        self.label_3 = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout_5.addWidget(self.label_3, 0, 0, 1, 1)
        self.label_stim_type = QtGui.QLabel(self.centralwidget)
        self.label_stim_type.setObjectName(_fromUtf8("label_stim_type"))
        self.gridLayout_5.addWidget(self.label_stim_type, 0, 1, 1, 1)
        self.textEdit = QtGui.QTextEdit(self.centralwidget)
        self.textEdit.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEdit.sizePolicy().hasHeightForWidth())
        self.textEdit.setSizePolicy(sizePolicy)
        self.textEdit.setMinimumSize(QtCore.QSize(150, 0))
        self.textEdit.setObjectName(_fromUtf8("textEdit"))
        self.gridLayout_5.addWidget(self.textEdit, 2, 0, 1, 2)
        self.line_6 = QtGui.QFrame(self.centralwidget)
        self.line_6.setFrameShape(QtGui.QFrame.HLine)
        self.line_6.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_6.setObjectName(_fromUtf8("line_6"))
        self.gridLayout_5.addWidget(self.line_6, 1, 0, 1, 2)
        self.horizontalLayout_2.addLayout(self.gridLayout_5)
        spacerItem3 = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName(_fromUtf8("gridLayout_3"))
        self.pushButton_historgram = QtGui.QPushButton(self.centralwidget)
        self.pushButton_historgram.setObjectName(_fromUtf8("pushButton_historgram"))
        self.gridLayout_3.addWidget(self.pushButton_historgram, 1, 0, 1, 1)
        self.pushButton_tuning_curve = QtGui.QPushButton(self.centralwidget)
        self.pushButton_tuning_curve.setObjectName(_fromUtf8("pushButton_tuning_curve"))
        self.gridLayout_3.addWidget(self.pushButton_tuning_curve, 2, 0, 1, 1)
        self.pushButton_raster = QtGui.QPushButton(self.centralwidget)
        self.pushButton_raster.setStatusTip(_fromUtf8(""))
        self.pushButton_raster.setWhatsThis(_fromUtf8(""))
        self.pushButton_raster.setObjectName(_fromUtf8("pushButton_raster"))
        self.gridLayout_3.addWidget(self.pushButton_raster, 0, 0, 1, 1)
        self.pushButton_io_test = QtGui.QPushButton(self.centralwidget)
        self.pushButton_io_test.setObjectName(_fromUtf8("pushButton_io_test"))
        self.gridLayout_3.addWidget(self.pushButton_io_test, 3, 0, 1, 1)
        self.pushButton_spike_rates = QtGui.QPushButton(self.centralwidget)
        self.pushButton_spike_rates.setObjectName(_fromUtf8("pushButton_spike_rates"))
        self.gridLayout_3.addWidget(self.pushButton_spike_rates, 4, 0, 1, 1)
        self.pushButton_abr = QtGui.QPushButton(self.centralwidget)
        self.pushButton_abr.setObjectName(_fromUtf8("pushButton_abr"))
        self.gridLayout_3.addWidget(self.pushButton_abr, 5, 0, 1, 1)
        self.horizontalLayout_2.addLayout(self.gridLayout_3)
        self.gridLayout.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 681, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QObject.connect(self.checkBox_custom_window, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.label_xmax.setEnabled)
        QtCore.QObject.connect(self.checkBox_custom_window, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.doubleSpinBox_xmax.setEnabled)
        QtCore.QObject.connect(self.checkBox_custom_window, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.doubleSpinBox_xmin.setEnabled)
        QtCore.QObject.connect(self.checkBox_custom_window, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.label_xmin.setEnabled)
        QtCore.QObject.connect(self.checkBox_custom_window, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.label_ymax.setEnabled)
        QtCore.QObject.connect(self.checkBox_custom_window, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.doubleSpinBox_ymax.setEnabled)
        QtCore.QObject.connect(self.checkBox_custom_window, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.doubleSpinBox_ymin.setEnabled)
        QtCore.QObject.connect(self.checkBox_custom_window, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.label_ymin.setEnabled)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Sparkle Analysis", None))
        self.label.setText(_translate("MainWindow", "Open:", None))
        self.pushButton_browse.setText(_translate("MainWindow", "Browse...", None))
        self.label_comments.setText(_translate("MainWindow", "Comments:", None))
        self.label_test_num.setText(_translate("MainWindow", "Test Number:", None))
        self.label_trace.setText(_translate("MainWindow", "Trace:", None))
        self.label_rep.setText(_translate("MainWindow", "Repetition:", None))
        self.label_channel.setText(_translate("MainWindow", "Channel:", None))
        self.Title.setText(_translate("MainWindow", "Sparkle Analysis", None))
        self.checkBox_custom_window.setText(_translate("MainWindow", "Use Custom Window", None))
        self.label_xmax.setText(_translate("MainWindow", "X Max", None))
        self.doubleSpinBox_xmax.setSuffix(_translate("MainWindow", " s", None))
        self.doubleSpinBox_xmin.setSuffix(_translate("MainWindow", " s", None))
        self.label_xmin.setText(_translate("MainWindow", "X Min", None))
        self.label_ymax.setText(_translate("MainWindow", "Y Max", None))
        self.doubleSpinBox_ymax.setSuffix(_translate("MainWindow", " V", None))
        self.doubleSpinBox_ymin.setSuffix(_translate("MainWindow", " V", None))
        self.label_ymin.setText(_translate("MainWindow", "Y Min", None))
        self.label_2.setText(_translate("MainWindow", "Threshold:", None))
        self.doubleSpinBox_threshold.setSuffix(_translate("MainWindow", " V", None))
        self.pushButton_auto_threshold.setText(_translate("MainWindow", "Estimate Threshold", None))
        self.label_5.setText(_translate("MainWindow", "Response Polarity:", None))
        self.radioButton_normal.setText(_translate("MainWindow", "Normal", None))
        self.radioButton_inverse.setText(_translate("MainWindow", "Inverse", None))
        self.label_3.setText(_translate("MainWindow", "Stim Type:", None))
        self.label_stim_type.setText(_translate("MainWindow", "None", None))
        self.pushButton_historgram.setToolTip(_translate("MainWindow", "Creates a histogram based  on your test number, trace and threshold.", None))
        self.pushButton_historgram.setText(_translate("MainWindow", "Histogram", None))
        self.pushButton_tuning_curve.setToolTip(_translate("MainWindow", "Opens a window for creating tuning curve plots.", None))
        self.pushButton_tuning_curve.setText(_translate("MainWindow", "Tuning Curve", None))
        self.pushButton_raster.setToolTip(_translate("MainWindow", "Creates a raster based  on your test number, trace and threshold.", None))
        self.pushButton_raster.setText(_translate("MainWindow", "Raster", None))
        self.pushButton_io_test.setToolTip(_translate("MainWindow", "Opens a window for creating I/O plots.", None))
        self.pushButton_io_test.setText(_translate("MainWindow", "I/O Test", None))
        self.pushButton_spike_rates.setToolTip(_translate("MainWindow", "Opens a window for creating spike rate plots.", None))
        self.pushButton_spike_rates.setText(_translate("MainWindow", "Spike Rates", None))
        self.pushButton_abr.setToolTip(_translate("MainWindow", "Creates an auditory brainstem response (ABR) graph based on your current test.", None))
        self.pushButton_abr.setText(_translate("MainWindow", "ABR", None))

from util.pyqtgraph_widgets import TraceWidget
