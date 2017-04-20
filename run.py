import os
import sys
import csv
import h5py
import operator
import numpy as np
import seaborn as sns
import scipy.stats as stats

import pandas as pd
import matplotlib.pyplot as plt

from PyQt4 import QtCore, QtGui
from ui.main_ui import Ui_MainWindow

from dialogs.SpikeRatesDialog import SpikeRatesDialog
from dialogs.MultiIODialog import MultiIODialog
from dialogs.ABRDialog import ABRDialog
from dialogs.TuningCurveDialog import TuningCurveDialog

from util.spikerates import ResponseStats
from util.spikerates import ResponseStatsSpikes

from util.spikestats import get_spike_times


class MyForm(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.dialog = QtGui.QMainWindow

        self.filename = ''
        self.threshold = 0

        self.message_num = 0

        self.ui.textEdit.setReadOnly(True)

        # TODO Complete Spike Rates
        # self.ui.pushButton_spike_rates.setEnabled(False)

        QtCore.QObject.connect(self.ui.pushButton_raster, QtCore.SIGNAL("clicked()"), self.graph_raster)
        QtCore.QObject.connect(self.ui.pushButton_historgram, QtCore.SIGNAL("clicked()"), self.graph_historgram)
        QtCore.QObject.connect(self.ui.pushButton_tuning_curve, QtCore.SIGNAL("clicked()"), self.graph_tuning_curve)
        # QtCore.QObject.connect(self.ui.pushButton_io_test, QtCore.SIGNAL("clicked()"), self.graph_io_test)
        QtCore.QObject.connect(self.ui.pushButton_io_test, QtCore.SIGNAL("clicked()"), self.graph_multi_io_test)
        QtCore.QObject.connect(self.ui.pushButton_spike_rates, QtCore.SIGNAL("clicked()"), self.graph_spike_rates)
        QtCore.QObject.connect(self.ui.pushButton_abr, QtCore.SIGNAL("clicked()"), self.graph_abr)

        QtCore.QObject.connect(self.ui.pushButton_browse, QtCore.SIGNAL("clicked()"), self.browse)
        QtCore.QObject.connect(self.ui.pushButton_auto_threshold, QtCore.SIGNAL("clicked()"), self.auto_threshold)
        QtCore.QObject.connect(self.ui.radioButton_normal, QtCore.SIGNAL("toggled(bool)"), self.generate_view)
        QtCore.QObject.connect(self.ui.doubleSpinBox_threshold, QtCore.SIGNAL("valueChanged(const QString&)"), self.update_thresh)

        QtCore.QObject.connect(self.ui.comboBox_test_num, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.load_traces)
        QtCore.QObject.connect(self.ui.comboBox_trace, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.load_reps)
        QtCore.QObject.connect(self.ui.comboBox_rep, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.load_channels)
        QtCore.QObject.connect(self.ui.comboBox_channel, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.generate_view)

        QtCore.QObject.connect(self.ui.checkBox_custom_window, QtCore.SIGNAL("toggled(bool)"), self.update_window)
        QtCore.QObject.connect(self.ui.doubleSpinBox_xmax, QtCore.SIGNAL("valueChanged(const QString&)"), self.update_window)
        QtCore.QObject.connect(self.ui.doubleSpinBox_xmin, QtCore.SIGNAL("valueChanged(const QString&)"), self.update_window)
        QtCore.QObject.connect(self.ui.doubleSpinBox_ymax, QtCore.SIGNAL("valueChanged(const QString&)"), self.update_window)
        QtCore.QObject.connect(self.ui.doubleSpinBox_ymin, QtCore.SIGNAL("valueChanged(const QString&)"), self.update_window)

        self.ui.view.sigRangeChanged.connect(self.update_view_range)
        self.ui.view.threshLine.sigPositionChangeFinished.connect(self.update_thresh2)

    def browse(self):
        self.ui.comboBox_test_num.clear()

        QtGui.QFileDialog(self)
        self.filename = QtGui.QFileDialog.getOpenFileName()
        self.ui.lineEdit_file_name.setText(self.filename)

        # If the filename is not blank, attempt to extract test numbers and place them into the combobox
        if self.filename != '':
            if '.hdf5' in self.filename:
                try:
                    h_file = h5py.File(unicode(self.filename), 'r')
                except IOError:
                    self.add_message('Error: I/O Error')
                    self.ui.label_test_num.setEnabled(False)
                    self.ui.label_comments.setEnabled(False)
                    self.ui.lineEdit_comments.setEnabled(False)
                    self.ui.lineEdit_comments.setText('')
                    self.ui.comboBox_test_num.setEnabled(False)
                    return

                tests = {}
                for key in h_file.keys():
                    if 'segment' in key:
                        for test in h_file[key].keys():
                            tests[test] = int(test.replace('test_', ''))

                sorted_tests = sorted(tests.items(), key=operator.itemgetter(1))

                for test in sorted_tests:
                    self.ui.comboBox_test_num.addItem(test[0])

                self.ui.label_test_num.setEnabled(True)
                self.ui.label_comments.setEnabled(True)
                self.ui.lineEdit_comments.setEnabled(True)
                self.ui.comboBox_test_num.setEnabled(True)

                h_file.close()

            else:
                self.add_message('Error: Must select a .hdf5 file.')
                self.ui.label_test_num.setEnabled(False)
                self.ui.label_comments.setEnabled(False)
                self.ui.lineEdit_comments.setEnabled(False)
                self.ui.lineEdit_comments.setText('')
                self.ui.comboBox_test_num.setEnabled(False)
                return
        else:
            self.add_message('Error: Must select a file to open.')
            self.ui.label_test_num.setEnabled(False)
            self.ui.label_comments.setEnabled(False)
            self.ui.lineEdit_comments.setEnabled(False)
            self.ui.lineEdit_comments.setText('')
            self.ui.comboBox_test_num.setEnabled(False)
            return

    def auto_threshold(self):
        thresh_fraction = 0.7

        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if self.valid_filename(filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            return

        # Find target segment
        for segment in h_file.keys():
            for test in h_file[segment].keys():
                if target_test == test:
                    target_seg = segment
                    target_test = test

        trace_data = h_file[target_seg][target_test].value

        if len(trace_data.shape) == 4:
            trace_data = trace_data.squeeze()

        # Still shape of 4
        if len(trace_data.shape) == 4:
            target_chan = int(self.ui.comboBox_channel.currentText().replace('channel_', '')) - 1
            trace_data = trace_data[:, :, target_chan, :]
            trace_data = trace_data.squeeze()

        # Compute threshold from average maximum of traces
        max_trace = []
        for n in range(len(trace_data[1, :, 0])):
            max_trace.append(np.max(np.abs(trace_data[1, n, :])))
        average_max = np.array(max_trace).mean()
        thresh = thresh_fraction * average_max

        self.ui.doubleSpinBox_threshold.setValue(thresh)
        self.update_thresh()

        h_file.close()

    def generate_view(self):
        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if self.valid_filename(filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            return

        # clear view
        self.clear_view()

        if self.ui.radioButton_normal.isChecked():
            self.ui.view.invertPolarity(False)

        if self.ui.radioButton_inverse.isChecked():
            self.ui.view.invertPolarity(True)

        for key in h_file.keys():
            if 'segment' in key:
                for test in h_file[key].keys():
                    if target_test == test:
                        target_seg = key
                        target_test = test

        # Makes the assumption that all of the traces are of the same type
        stim_info = eval(h_file[target_seg][target_test].attrs['stim'])
        self.ui.label_stim_type.setText(stim_info[1]['components'][0]['stim_type'])

        fs = h_file[target_seg].attrs['samplerate_ad']

        target_trace = []
        target_rep = []
        target_chan = []

        # Get the values from the combo boxes
        self.ui.comboBox_test_num.currentText()
        if self.ui.comboBox_trace.currentText() != '':
            target_trace = int(self.ui.comboBox_trace.currentText().replace('trace_', '')) - 1
        if self.ui.comboBox_rep.currentText() != 'All' and self.ui.comboBox_rep.currentText() != '':
            target_rep = int(self.ui.comboBox_rep.currentText().replace('rep_', '')) - 1
        if self.ui.comboBox_channel.currentText() != '':
            target_chan = int(self.ui.comboBox_channel.currentText().replace('channel_', '')) - 1

        test_data = h_file[target_seg][target_test].value

        # Get the presentation data depending on if there is a channel field or not
        if len(test_data.shape) == 4:
            presentation = test_data[target_trace, target_rep, target_chan, :]
        elif len(test_data.shape) == 3:
            presentation = test_data[target_trace, target_rep, :]

        len_presentation = len(presentation)

        # Get the length of the window and length of presentation depending on if all is selected or not
        if len_presentation != 0:
            window = len(presentation) / float(fs)
        else:
            if len(test_data.shape) == 4:
                window = len(test_data[0, 0, 0, :]) / float(fs)
                len_presentation = len(test_data[0, 0, 0, :])
            elif len(test_data.shape) == 3:
                window = len(test_data[0, 0, :]) / float(fs)
                len_presentation = len(test_data[0, 0, :])

        xlist = np.linspace(0, float(window), len_presentation)
        ylist = presentation

        # If channel didn't load, set default as 0
        if not target_chan:
            target_chan = 0

        # Set window size
        if not self.ui.checkBox_custom_window.checkState():
            if len(presentation) > 0:
                ymin = min(presentation)
                ymax = max(presentation)
            else:
                ymin = 0
                ymax = 0
                if len(test_data.shape) == 3:
                    rep_len = test_data.shape[1]
                    for i in range(rep_len):
                        if not test_data[target_trace, i, :].any():
                            return
                        if min(test_data[target_trace, i, :]) < ymin:
                            ymin = min(test_data[target_trace, i, :])
                        if max(test_data[target_trace, i, :]) > ymax:
                            ymax = max(test_data[target_trace, i, :])
                else:
                    rep_len = test_data.shape[1]
                    for i in range(rep_len):
                        if not test_data[target_trace, i, target_chan, :].any():
                            return
                        if min(test_data[target_trace, i, target_chan, :]) < ymin:
                            ymin = min(test_data[target_trace, i, target_chan, :])
                        if max(test_data[target_trace, i, target_chan, :]) > ymax:
                            ymax = max(test_data[target_trace, i, target_chan, :])

            self.ui.view.setXRange(0, window, 0)
            self.ui.view.setYRange(ymin, ymax, 0.1)

        if self.ui.comboBox_rep.currentText() == 'All':
            self.ui.view.tracePlot.clear()
            # Fix xlist to be the length of presentation
            if len(test_data.shape) == 3:
                self.ui.view.addTraces(xlist, test_data[target_trace, :, :])
            else:
                self.ui.view.addTraces(xlist, test_data[target_trace, :, target_chan, :])
            self.ui.radioButton_normal.setChecked(True)
            self.ui.radioButton_normal.setEnabled(False)
            self.ui.radioButton_inverse.setEnabled(False)
        else:
            self.ui.view.updateData(axeskey='response', x=xlist, y=ylist)
            self.ui.radioButton_normal.setEnabled(True)
            self.ui.radioButton_inverse.setEnabled(True)

        h_file.close()

    def valid_filename(self, filename):
        # Validate filename
        if filename != '':
            if '.hdf5' in filename:
                try:
                    temp_file = h5py.File(unicode(self.filename), 'r')
                    temp_file.close()
                except IOError:
                    self.add_message('Error: I/O Error')
                    return False
            else:
                self.add_message('Error: Must select a .hdf5 file.')
                return False
        else:
            self.add_message('Error: Must select a file to open.')
            return False

        return True

    def load_traces(self):
        self.ui.comboBox_trace.clear()

        # Validate filename
        if self.valid_filename(self.filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            self.ui.label_trace.setEnabled(False)
            self.ui.comboBox_trace.setEnabled(False)
            return

        if self.ui.comboBox_test_num.currentText() == 'All' or self.ui.comboBox_test_num.currentText() == '':
            self.ui.label_trace.setEnabled(False)
            self.ui.comboBox_trace.setEnabled(False)
            self.ui.comboBox_trace.clear()
            h_file.close()
            return
        else:
            self.ui.label_trace.setEnabled(True)
            self.ui.comboBox_trace.setEnabled(True)

        for key in h_file.keys():
            if 'segment' in key:
                for test in h_file[key].keys():
                    if target_test == test:
                        target_seg = key
                        target_test = test

        traces = h_file[target_seg][target_test].value.shape[0]

        for i in range(traces):
            self.ui.comboBox_trace.addItem('trace_' + str(i+1))

        self.ui.label_trace.setEnabled(True)
        self.ui.comboBox_trace.setEnabled(True)

        try:
            comment = h_file[target_seg].attrs['comment']
            self.ui.lineEdit_comments.setText(comment)
        except:
            print 'Failed to load comment'

        h_file.close()

    def load_reps(self):
        self.ui.comboBox_rep.clear()

        # Validate filename
        if self.valid_filename(self.filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            self.ui.label_rep.setEnabled(False)
            self.ui.comboBox_rep.setEnabled(False)
            return

        if self.ui.comboBox_trace.currentText() == 'All' or self.ui.comboBox_trace.currentText() == '':
            self.ui.label_rep.setEnabled(False)
            self.ui.comboBox_rep.setEnabled(False)
            self.ui.comboBox_rep.clear()
            h_file.close()
            return
        else:
            self.ui.label_rep.setEnabled(True)
            self.ui.comboBox_rep.setEnabled(True)

        self.ui.comboBox_rep.addItem('All')

        for key in h_file.keys():
            if 'segment' in key:
                for test in h_file[key].keys():
                    if target_test == test:
                        target_seg = key
                        target_test = test

        reps = h_file[target_seg][target_test].value.shape[1]

        for i in range(reps):
            self.ui.comboBox_rep.addItem('rep_' + str(i+1))

        h_file.close()

    def load_channels(self):
        self.ui.comboBox_channel.clear()

        # Validate filename
        if self.valid_filename(self.filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            self.ui.label_rep.setEnabled(False)
            self.ui.comboBox_rep.setEnabled(False)
            return

        if self.ui.comboBox_test_num.count() == 0:
            h_file.close()
            self.clear_view()
            return

        for key in h_file.keys():
            if 'segment' in key:
                for test in h_file[key].keys():
                    if target_test == test:
                        target_seg = key
                        target_test = test

        if len(h_file[target_seg][target_test].value.shape) > 3:
            channels = h_file[target_seg][target_test].value.shape[2]
        else:
            channels = 1

        if channels == 1:
            self.ui.comboBox_channel.addItem('channel_1')
        else:
            for i in range(channels):
                self.ui.comboBox_channel.addItem('channel_' + str(i+1))

        if self.ui.comboBox_rep.currentText() == '' or self.ui.comboBox_channel.count() < 2:
            self.ui.label_channel.setEnabled(False)
            self.ui.comboBox_channel.setEnabled(False)
            self.ui.comboBox_channel.clear()
            self.ui.comboBox_channel.addItem('channel_1')
        else:
            self.ui.label_channel.setEnabled(True)
            self.ui.comboBox_channel.setEnabled(True)

        if self.ui.comboBox_trace.currentText() != '' and self.ui.comboBox_rep.currentText() != '' and self.ui.comboBox_channel != '':
            self.generate_view()

        h_file.close()

    def update_thresh(self):
        self.ui.view.setThreshold(self.ui.doubleSpinBox_threshold.value())
        self.ui.view.update_thresh()

    def update_thresh2(self):
        self.ui.doubleSpinBox_threshold.setValue(self.ui.view.getThreshold())

    def update_window(self):
        if self.ui.checkBox_custom_window.checkState():
            self.ui.view.setXRange(self.ui.doubleSpinBox_xmin.value(), self.ui.doubleSpinBox_xmax.value(), 0)
            self.ui.view.setYRange(self.ui.doubleSpinBox_ymin.value(), self.ui.doubleSpinBox_ymax.value(), 0)

    def update_view_range(self):
        view_range = self.ui.view.visibleRange()
        self.ui.doubleSpinBox_xmin.setValue(view_range.left())
        self.ui.doubleSpinBox_xmax.setValue(view_range.right())
        self.ui.doubleSpinBox_ymin.setValue(view_range.top())
        self.ui.doubleSpinBox_ymax.setValue(view_range.bottom())

    def count_spikes(self):
        pass

    def generate_rasters(self):
        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if not self.valid_filename(filename):
            return

        target_test = self.ui.comboBox_test_num.currentText()
        thresh = self.ui.doubleSpinBox_threshold.value()

        h_file = h5py.File(unicode(filename), 'r')

        # Find target segment
        for segment in h_file.keys():
            for test in h_file[segment].keys():
                if target_test == test:
                    target_seg = segment
                    target_test = test

        fs = h_file[target_seg].attrs['samplerate_ad']
        reps = h_file[target_seg][target_test].attrs['reps']
        start_time = h_file[target_seg][target_test].attrs['start']
        trace_data = h_file[target_seg][target_test].value

        if len(trace_data.shape) == 4:
            pass

        stim_info = eval(h_file[target_seg][target_test].attrs['stim'])
        duration = trace_data.shape[-1] / fs * 1000
        if stim_info[1]['components'][0]['stim_type'] != 'Pure Tone':
            self.add_message('Cannot generate raster with stim type "' + str(stim_info[1]['components'][0]['stim_type']) + '".')
            h_file.close()
            return

        autoRasters = {}
        for tStim in range(1, len(stim_info)):

            freq = int(stim_info[tStim]['components'][0]['frequency'])
            spl = int(stim_info[tStim]['components'][0]['intensity'])
            trace_key = str(freq) + '_' + str(spl)
            num_reps = trace_data.shape[1]  # ???: Same as reps = h_file[target_seg][target_test].attrs['reps']
            spikeTrains = pd.DataFrame([])
            nspk = 0
            for tRep in range(reps):

                if len(trace_data.shape) == 3:
                    trace = trace_data[tStim][tRep]
                    pass
                elif len(trace_data.shape) == 4:
                    tchan = int(self.ui.comboBox_channel.currentText().replace('channel_', '')) - 1
                    trace = trace_data[tStim][tRep][tchan]
                    pass
                else:
                    self.add_message('Cannot handle trace_data of shape: ' + str(trace_data.shape))
                    return

                spike_times = 1000 * np.array(get_spike_times(trace, thresh, fs))
                spike_times_s = pd.Series(spike_times)

                if spike_times_s.size > nspk:
                    spikeTrains = spikeTrains.reindex(spike_times_s.index)
                    nspk = spike_times_s.size
                spikeTrains[str(tRep)] = spike_times_s
                autoRasters[trace_key] = spikeTrains

        h_file.close()

        return autoRasters

    def graph_spike_rates(self):
        self.dialog = SpikeRatesDialog()

        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if not self.valid_filename(filename):
            return

        self.dialog.populate_checkboxes(filename)
        self.dialog.setWindowIcon(QtGui.QIcon('images/horsey.ico'))
        self.dialog.setWindowTitle('Sparkle Analysis - Spike Rates')
        self.dialog.show()

    def graph_multi_io_test(self):
        self.dialog = MultiIODialog()

        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if not self.valid_filename(filename):
            return

        self.dialog.populate_checkboxes(filename, self.ui.doubleSpinBox_threshold.value())
        self.dialog.setWindowIcon(QtGui.QIcon('images/horsey.ico'))
        self.dialog.setWindowTitle('Sparkle Analysis - I/O Test')
        self.dialog.show()

    def graph_abr(self):
        self.dialog = ABRDialog()

        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if not self.valid_filename(filename):
            return

        self.dialog.populate_boxes(filename)
        self.dialog.setWindowIcon(QtGui.QIcon('images/horsey.ico'))
        self.dialog.setWindowTitle('Sparkle Analysis - ABRs')
        self.dialog.show()

    def graph_tuning_curve(self):
        self.dialog = TuningCurveDialog()

        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if not self.valid_filename(filename):
            return

        self.dialog.populate_boxes(filename)
        self.dialog.set_threshold(self.ui.doubleSpinBox_threshold.value())
        self.dialog.set_test_num(self.ui.comboBox_test_num.currentIndex())
        self.dialog.set_channel_num(self.ui.comboBox_channel.currentIndex())
        self.dialog.set_trace_num(self.ui.comboBox_trace.currentIndex())
        self.dialog.setWindowIcon(QtGui.QIcon('images/horsey.ico'))
        self.dialog.setWindowTitle('Sparkle Analysis - Tuning Curve Generator')
        self.dialog.show()

    def graph_io_test(self):
        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if not self.valid_filename(filename):
            return

        target_test = self.ui.comboBox_test_num.currentText()
        thresh = self.ui.doubleSpinBox_threshold.value()

        h_file = h5py.File(unicode(filename), 'r')

        # Find target segment
        for segment in h_file.keys():
            for test in h_file[segment].keys():
                if target_test == test:
                    target_seg = segment
                    target_test = test

        fs = h_file[target_seg].attrs['samplerate_ad']
        reps = h_file[target_seg][target_test].attrs['reps']
        start_time = h_file[target_seg][target_test].attrs['start']
        trace_data = h_file[target_seg][target_test].value

        if len(trace_data.shape) == 4:
            pass

        stim_info = eval(h_file[target_seg][target_test].attrs['stim'])
        duration = trace_data.shape[-1] / fs * 1000

        autoRasters = {}
        for tStim in range(1, len(stim_info)):
            spl = int(stim_info[tStim]['components'][0]['intensity'])
            traceKey = 'None_' + str(spl).zfill(2)
            spikeTrains = pd.DataFrame([])
            nspk = 0
            for tRep in range(reps):

                if len(trace_data.shape) == 3:
                    trace = trace_data[tStim][tRep]
                    pass
                elif len(trace_data.shape) == 4:
                    tchan = int(self.ui.comboBox_channel.currentText().replace('channel_', '')) - 1
                    trace = trace_data[tStim][tRep][tchan]
                    pass
                else:
                    self.add_message('Cannot handle trace_data of shape: ' + str(trace_data.shape))
                    return

                spike_times = 1000 * np.array(get_spike_times(trace, thresh, fs))
                spike_times_s = pd.Series(spike_times)

                if spike_times_s.size >= nspk:
                    spikeTrains = spikeTrains.reindex(spike_times_s.index)
                    nspk = spike_times_s.size
                spikeTrains[str(tRep)] = spike_times_s
            autoRasters[traceKey] = spikeTrains
        rasters = autoRasters

        h_file.close()

        tuning = []

        sortedKeys = sorted(rasters.keys())
        for traceKey in sortedKeys:
            spl = int(traceKey.split('_')[-1])
            raster = rasters[traceKey]
            res = ResponseStats(raster)
            tuning.append({'intensity': spl, 'response': res[0], 'responseSTD': res[1]})

        tuningCurves = pd.DataFrame(tuning)
        tuningCurves.plot(x='intensity', y='response', yerr='responseSTD', capthick=1, label=str(target_test))
        plt.legend(loc='upper left', fontsize=12, frameon=True)
        sns.despine()
        plt.grid(False)
        plt.xlabel('Intensity (dB)', size=14)
        plt.ylabel('Response Rate (Hz)', size=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.title(str.split(str(filename), '/')[-1].replace('.hdf5', '') + ' '
                  + str(self.ui.comboBox_test_num.currentText()).replace('test_', 'Test '))

        plt.show()

    def graph_raster(self):
        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if not self.valid_filename(filename):
            return

        target_test = self.ui.comboBox_test_num.currentText()
        thresh = self.ui.doubleSpinBox_threshold.value()

        h_file = h5py.File(unicode(filename), 'r')

        # Find target segment
        for segment in h_file.keys():
            for test in h_file[segment].keys():
                if target_test == test:
                    target_seg = segment
                    target_test = test

        target_trace = int(self.ui.comboBox_trace.currentText().replace('trace_', '')) - 1

        fs = h_file[target_seg].attrs['samplerate_ad']
        reps = h_file[target_seg][target_test].attrs['reps']
        start_time = h_file[target_seg][target_test].attrs['start']
        trace_data = h_file[target_seg][target_test].value

        if len(trace_data.shape) == 4:
            pass

        stim_info = eval(h_file[target_seg][target_test].attrs['stim'])
        duration = trace_data.shape[-1] / fs * 1000

        autoRasters = {}
        tStim = target_trace

        spikeTrains = pd.DataFrame([])
        nspk = 0
        for tRep in range(reps):

            if len(trace_data.shape) == 3:
                trace = trace_data[tStim][tRep]
                pass
            elif len(trace_data.shape) == 4:
                tchan = int(self.ui.comboBox_channel.currentText().replace('channel_', '')) - 1
                trace = trace_data[tStim][tRep][tchan]
                pass
            else:
                self.add_message('Cannot handle trace_data of shape: ' + str(trace_data.shape))
                h_file.close()
                return

            spike_times = 1000 * np.array(get_spike_times(trace, thresh, fs))
            spike_times_s = pd.Series(spike_times)

            if spike_times_s.size > nspk:
                spikeTrains = spikeTrains.reindex(spike_times_s.index)
                nspk = spike_times_s.size
            spikeTrains[str(tRep)] = spike_times_s
        rasters = spikeTrains

        h_file.close()

        if len(rasters.shape) > 1:
            spks = np.array([])
            trns = np.array([])
            for trnNum in range(len(rasters.columns)):
                spkTrn = np.array(rasters.iloc[:, trnNum].dropna())
                trns = np.hstack([trns, (trnNum + 1) * np.ones(len(spkTrn))])
                spks = np.hstack([spks, spkTrn])
            # --- Raster plot of spikes ---
            sns.set_style("white")
            sns.set_style("ticks")
            raster_f = plt.figure(figsize=(8, 2))
            sns.despine()
            plt.grid(False)
            ax = plt.scatter(spks, trns, marker='s', s=5, color='k')
            plt.ylim(len(rasters.columns) + 0.5, 0.5)
            plt.xlim(0, duration)
            plt.xlabel('Time (ms)')
            plt.ylabel('Presentation cycle')
            plt.title(str.split(str(filename), '/')[-1].replace('.hdf5', '') + ' '
                  + str(self.ui.comboBox_test_num.currentText()).replace('test_', 'Test '))
            plt.tick_params(axis='both', which='major', labelsize=14)

            raster_f.show()
        else:
            self.add_message('Only spike timing information provided, requires presentation numbers for raster.')

        return ax

    def graph_historgram(self):
        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if not self.valid_filename(filename):
            return

        target_test = self.ui.comboBox_test_num.currentText()
        thresh = self.ui.doubleSpinBox_threshold.value()

        h_file = h5py.File(unicode(filename), 'r')

        # Find target segment
        for segment in h_file.keys():
            for test in h_file[segment].keys():
                if target_test == test:
                    target_seg = segment
                    target_test = test

        target_trace = int(self.ui.comboBox_trace.currentText().replace('trace_', '')) - 1

        fs = h_file[target_seg].attrs['samplerate_ad']
        reps = h_file[target_seg][target_test].attrs['reps']
        start_time = h_file[target_seg][target_test].attrs['start']
        trace_data = h_file[target_seg][target_test].value

        if len(trace_data.shape) == 4:
            pass

        stim_info = eval(h_file[target_seg][target_test].attrs['stim'])
        duration = trace_data.shape[-1] / fs * 1000

        autoRasters = {}
        tStim = target_trace

        spikeTrains = pd.DataFrame([])
        nspk = 0
        for tRep in range(reps):

            if len(trace_data.shape) == 3:
                trace = trace_data[tStim][tRep]
                pass
            elif len(trace_data.shape) == 4:
                tchan = int(self.ui.comboBox_channel.currentText().replace('channel_', '')) - 1
                trace = trace_data[tStim][tRep][tchan]
                pass
            else:
                self.add_message('Cannot handle trace_data of shape: ' + str(trace_data.shape))
                return

            spike_times = 1000 * np.array(get_spike_times(trace, thresh, fs))
            spike_times_s = pd.Series(spike_times)

            if spike_times_s.size > nspk:
                spikeTrains = spikeTrains.reindex(spike_times_s.index)
                nspk = spike_times_s.size
            spikeTrains[str(tRep)] = spike_times_s
        rasters = spikeTrains

        h_file.close()

        if len(rasters.shape) > 1:
            spks = np.array([])
            trns = np.array([])
            for trnNum in range(len(rasters.columns)):
                spkTrn = np.array(rasters.iloc[:, trnNum].dropna())
                trns = np.hstack([trns, (trnNum + 1) * np.ones(len(spkTrn))])
                spks = np.hstack([spks, spkTrn])
            spikeTimes = rasters.stack()
        else:
            spikeTimes = rasters.dropna()
        # --- Histogram of spike times (1 ms bins)---
        sns.set_style("white")
        sns.set_style("ticks")
        histogram_f = plt.figure(figsize=(8, 3))
        axHist = spikeTimes.hist(bins=int(duration / 1), range=(0, duration))  # , figsize=(8,3))
        sns.despine()
        plt.xlim(0, duration)
        plt.xlabel('Time (ms)', size=14)
        plt.ylabel('Number of spikes', size=14)
        plt.title(str.split(str(filename), '/')[-1].replace('.hdf5', '') + ' '
                  + str(self.ui.comboBox_test_num.currentText()).replace('test_', 'Test '))
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.grid(False)
        histogram_f.show()

        return axHist

    def graph_abr_old(self):
        # Assumes there will only be one channel

        print 'Graphing ABR'
        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if not self.valid_filename(filename):
            return

        # clear view
        self.clear_view()

        target_test = self.ui.comboBox_test_num.currentText()

        h_file = h5py.File(unicode(filename), 'r')

        # Find target segment
        for segment in h_file.keys():
            for test in h_file[segment].keys():
                if target_test == test:
                    target_seg = segment
                    target_test = test

        # Makes the assumption that all of the traces are of the same type
        stim_info = eval(h_file[target_seg][target_test].attrs['stim'])
        self.ui.label_stim_type.setText(stim_info[1]['components'][0]['stim_type'])

        fs = h_file[target_seg].attrs['samplerate_ad']

        trace_data = h_file[target_seg][target_test].value

        if len(trace_data.shape) == 4:
            pass

        samples = trace_data.shape[-1]
        traces = trace_data.shape[0]
        reps = trace_data.shape[1]

        average = np.empty(samples)
        abr = np.empty(shape=(traces, 1, 1, samples))

        intensity = []

        # Average the rep data into one trace
        if len(trace_data.shape) == 4:
            for t in range(traces):
                for s in range(samples):
                    # print s
                    for r in range(reps):
                        # print r
                        average[s] += trace_data[0, r, 0, s]
                    average[s] /= reps + 1
                    abr[t, 0, 0, s] = average[s]
                intensity.append(stim_info[t]['components'][0]['intensity'])
        elif len(trace_data.shape) == 3:
            for t in range(traces):
                for s in range(samples):
                    # print s
                    for r in range(reps):
                        # print r
                        average[s] += trace_data[0, r, s]
                    average[s] /= reps + 1
                    abr[t, 0, 0, s] = average[s]
                intensity.append(stim_info[t]['components'][0]['intensity'])
        else:
            self.add_message('Cannot handle trace_data of shape: ' + str(trace_data.shape))
            return

        self.ui.view.tracePlot.clear()

        # Get the presentation data depending on if there is a channel field or not
        if len(abr.shape) == 4:
            presentation = abr[[], [], [], :]
        elif len(abr.shape) == 3:
            presentation = abr[[], [], :]

        len_presentation = len(presentation)

        # Get the length of the window and length of presentation depending on if all is selected or not
        if len_presentation != 0:
            window = len(presentation) / float(fs)
        else:
            if len(abr.shape) == 4:
                window = len(abr[0, 0, 0, :]) / float(fs)
                len_presentation = len(abr[0, 0, 0, :])
            elif len(abr.shape) == 3:
                window = len(abr[0, 0, :]) / float(fs)
                len_presentation = len(abr[0, 0, :])

        xlist = np.linspace(0, float(window), len_presentation)
        ylist = presentation

        # Fix xlist to be the length of presentation
        if len(abr.shape) == 3:
            self.ui.view.addTracesABR(xlist, abr[:, 0, :], intensity)
        else:
            self.ui.view.addTracesABR(xlist, abr[:, 0, 0, :], intensity)

        self.ui.radioButton_normal.setChecked(True)
        self.ui.radioButton_normal.setEnabled(False)
        self.ui.radioButton_inverse.setEnabled(False)

    def GetFreqsAttns(self, freqTuningHisto):  # Frequency Tuning Curve method
        """ Helper method for ShowSTH() to organize the frequencies in ascending order separated for each intensity.
        :param freqTuningHisto: dict of pandas.DataFrames with spike data
        :type freqTuningHisto: str
        :returns: ordered list of frequencies (DataFrame keys())
                  numpy array of frequencies
                  numpy array of intensities
        """
        freqs = np.array([])
        attns = np.array([])
        for histoKey in list(freqTuningHisto):
            if histoKey != 'None_None':
                freq = histoKey.split('_')[0]
                freqs = np.hstack([freqs, float(freq) / 1000])
                attn = histoKey.split('_')[1]
                attns = np.hstack([attns, float(attn)])
        attnCount = stats.itemfreq(attns)
        freqs = np.unique(freqs)
        attns = np.unique(attns)
        if np.max(attnCount[:, 1]) != np.min(attnCount[:, 1]):
            abortedAttnIdx = np.where(attnCount[:, 1] != np.max(attnCount[:, 1]))
            attns = np.delete(attns, abortedAttnIdx)
        orderedKeys = []
        for attn in attns:
            freqList = []
            for freq in freqs:
                key = str(int(freq * 1000)) + '_' + str(int(attn))
                freqList.append(key)
            orderedKeys.append(freqList)
        return orderedKeys, freqs, attns

    def add_message(self, message):
        self.message_num += 1
        self.ui.textEdit.append('[' + str(self.message_num) + ']: ' + message + '\n')

    def clear_view(self):
        self.ui.view.clearTraces()
        self.ui.view.clearMouse()
        self.ui.view.clearFocus()
        self.ui.view.clearMask()
        self.ui.view.clearData(axeskey='response')
        self.ui.view.tracePlot.clear()
        self.ui.view.rasterPlot.clear()
        self.ui.view.stimPlot.clear()
        self.ui.view.trace_stash = []


def check_output_folders():
    if not os.path.exists('output'):
        os.makedirs('output')

    # if not os.path.exists('output' + os.sep + 'rasters'):
    #     os.makedirs('output' + os.sep + 'rasters')
    #
    # if not os.path.exists('output' + os.sep + 'histograms'):
    #     os.makedirs('output' + os.sep + 'histograms')

    if not os.path.exists('output' + os.sep + 'tuning_curves'):
        os.makedirs('output' + os.sep + 'tuning_curves')


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myApp = MyForm()
    myApp.setWindowIcon(QtGui.QIcon('images/horsey.ico'))
    myApp.show()
    sys.exit(app.exec_())
