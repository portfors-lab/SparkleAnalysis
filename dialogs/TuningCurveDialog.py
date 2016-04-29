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

from ui.tuning_curves_ui import Ui_Form_tuning_curves

from util.spikerates import ResponseStats
from util.spikerates import ResponseStatsSpikes

from util.spikestats import get_spike_times


class TuningCurveDialog(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        self.ui = Ui_Form_tuning_curves()
        self.ui.setupUi(self)

        self.filename = ''

        self.message_num = 0

        # TODO Rename and add functionality in the future
        self.ui.radioButtonOther.setEnabled(False)

        QtCore.QObject.connect(self.ui.comboBox_test_num, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.populate_comments)
        QtCore.QObject.connect(self.ui.pushButton_auto_threshold, QtCore.SIGNAL("clicked()"), self.auto_threshold)
        QtCore.QObject.connect(self.ui.doubleSpinBox_threshold, QtCore.SIGNAL("valueChanged(const QString&)"), self.update_thresh)

        QtCore.QObject.connect(self.ui.comboBox_test_num, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.load_traces)
        QtCore.QObject.connect(self.ui.comboBox_trace, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.load_channels)

        QtCore.QObject.connect(self.ui.pushButtonGenerate, QtCore.SIGNAL("clicked()"), self.generate_tuning_curve)

    def set_threshold(self, thresh):
        self.ui.doubleSpinBox_threshold.setValue(thresh)

    def set_test_num(self, test_num):
        self.ui.comboBox_test_num.setCurrentIndex(test_num)

    def set_channel_num(self, chan_num):
        self.ui.comboBox_channel.setCurrentIndex(chan_num)

    def set_trace_num(self, trace_num):
        self.ui.comboBox_trace.setCurrentIndex(trace_num)

    def populate_boxes(self, filename):
        self.filename = filename

        self.ui.label_title.setText('Tuning Curve Generator - ' + str.split(str(filename), '/')[-1])

        self.ui.comboBox_test_num.clear()

        # If the filename is not blank, attempt to extract test numbers and place them into the combobox
        if self.filename != '':
            if '.hdf5' in self.filename:
                try:
                    h_file = h5py.File(unicode(self.filename), 'r')
                except IOError:
                    self.add_message('Error: I/O Error')
                    self.ui.lineEdit_comments.setEnabled(False)
                    self.ui.lineEdit_comments.setText('')
                    self.ui.comboBox_test_num.setEnabled(False)
                    self.ui.groupBoxPlot.setEnabled(False)
                    self.ui.groupBoxUnits.setEnabled(False)
                    return

                tests = {}
                for key in h_file.keys():
                    if 'segment' in key:
                        for test in h_file[key].keys():
                            tests[test] = int(test.replace('test_', ''))

                sorted_tests = sorted(tests.items(), key=operator.itemgetter(1))

                for test in sorted_tests:
                    self.ui.comboBox_test_num.addItem(test[0])

                self.ui.lineEdit_comments.setEnabled(True)
                self.ui.comboBox_test_num.setEnabled(True)
                self.ui.groupBoxPlot.setEnabled(True)
                self.ui.groupBoxUnits.setEnabled(True)

                h_file.close()

    def populate_comments(self):
        # Validate filename
        if self.valid_filename(self.filename):
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

        comment = h_file[target_seg].attrs['comment']
        self.ui.lineEdit_comments.setText(comment)

        # Populate stimulus info
        # Makes the assumption that all of the traces are of the same type
        stim_info = eval(h_file[target_seg][target_test].attrs['stim'])
        self.ui.label_stim_type.setText(stim_info[1]['components'][0]['stim_type'])

        h_file.close()

    def load_traces(self):
        self.ui.comboBox_trace.clear()

        # Validate filename
        if self.valid_filename(self.filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            self.ui.comboBox_trace.setEnabled(False)
            return

        if self.ui.comboBox_test_num.currentText() == 'All' or self.ui.comboBox_test_num.currentText() == '':
            self.ui.comboBox_trace.setEnabled(False)
            self.ui.comboBox_trace.clear()
            h_file.close()
            return
        else:
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

        self.ui.comboBox_trace.setEnabled(True)

        comment = h_file[target_seg].attrs['comment']
        self.ui.lineEdit_comments.setText(comment)

        h_file.close()

    def load_channels(self):
        self.ui.comboBox_channel.clear()

        # Validate filename
        if self.valid_filename(self.filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            self.ui.comboBox_channel.setEnabled(False)
            return

        for key in h_file.keys():
            if 'segment' in key:
                for test in h_file[key].keys():
                    if target_test == test:
                        target_seg = key
                        target_test = test

        if self.ui.comboBox_test_num.currentText() == '' or self.ui.comboBox_channel.count() < 2:
            self.ui.comboBox_channel.setEnabled(False)
            self.ui.comboBox_channel.clear()
        else:
            self.ui.comboBox_channel.setEnabled(True)

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

        if self.ui.comboBox_test_num.currentText() != '' and self.ui.comboBox_trace.currentText() != '' and self.ui.comboBox_channel != '':
            self.generate_view()

        h_file.close()

    def generate_view(self):
        filename = self.filename

        # Validate filename
        if self.valid_filename(filename):
            h_file = h5py.File(unicode(self.filename), 'r')
            target_test = self.ui.comboBox_test_num.currentText()
        else:
            return

        # Clear view
        self.clear_view()

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
        target_chan = []

        # Get the values from the combo boxes
        self.ui.comboBox_test_num.currentText()
        if self.ui.comboBox_trace.currentText() != '':
            target_trace = int(self.ui.comboBox_trace.currentText().replace('trace_', '')) - 1
        if self.ui.comboBox_channel.currentText() != '':
            target_chan = int(self.ui.comboBox_channel.currentText().replace('channel_', '')) - 1

        test_data = h_file[target_seg][target_test].value

        # Get the presentation data depending on if there is a channel field or not
        if len(test_data.shape) == 4:
            presentation = test_data[target_trace, [], target_chan, :]
        elif len(test_data.shape) == 3:
            presentation = test_data[target_trace, [], :]

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

        # Set window size
        if not self.ui.groupBoxWindow.isChecked():
            if len(presentation) > 0:
                ymin = min(presentation)
                ymax = max(presentation)
            else:
                ymin = 0
                ymax = 0
                if len(test_data.shape) == 3:
                    rep_len = test_data.shape[1]
                    for i in range(rep_len):
                        if min(test_data[target_trace, i, :]) < ymin:
                            ymin = min(test_data[target_trace, i, :])
                        if max(test_data[target_trace, i, :]) > ymax:
                            ymax = max(test_data[target_trace, i, :])
                else:
                    rep_len = test_data.shape[1]
                    for i in range(rep_len):
                        if min(test_data[target_trace, i, target_chan, :]) < ymin:
                            ymin = min(test_data[target_trace, i, target_chan, :])
                        if max(test_data[target_trace, i, target_chan, :]) > ymax:
                            ymax = max(test_data[target_trace, i, target_chan, :])

            self.ui.view.setXRange(0, window, 0)
            self.ui.view.setYRange(ymin, ymax, 0.1)

        # Display all repetitions
        self.ui.view.tracePlot.clear()
        # Fix xlist to be the length of presentation
        if len(test_data.shape) == 3:
            self.ui.view.addTraces(xlist, test_data[target_trace, :, :])
        else:
            self.ui.view.addTraces(xlist, test_data[target_trace, :, target_chan, :])

        h_file.close()

    def generate_tuning_curve(self):
        filename = self.filename

        # Validate filename
        if not self.valid_filename(filename):
            return

        rasters = self.generate_rasters()

        tuning = []
        orderedKeys, freqs, spls = self.GetFreqsAttns(rasters)

        for s in range(len(orderedKeys)):
            for k in orderedKeys[s]:
                freq = int(k.split('_')[0])
                spl = int(k.split('_')[1])
                raster = rasters[k]
                if self.ui.radioButtonMeanSpikes.isChecked():
                    res = ResponseStatsSpikes(raster)
                elif self.ui.radioButtonResponseRate.isChecked():
                    res = ResponseStats(raster)
                tuning.append({'intensity': spl, 'freq': freq / 1000, 'response': res[0], 'responseSTD': res[1]})

        tuningCurves = pd.DataFrame(tuning)

        if self.ui.radioButtonFreq.isChecked():
            # Rainbow Tuning Curve
            db = np.unique(tuningCurves['intensity'])

            axes = []
            for d in db:
                if axes:
                    tuningCurves[tuningCurves['intensity'] == d].plot(x='freq', y='response', ax=axes, yerr='responseSTD', capthick=1,
                                                                  label=str(d) + ' dB')
                else:
                    axes = tuningCurves[tuningCurves['intensity'] == d].plot(x='freq', y='response', yerr='responseSTD', capthick=1,
                                                                  label=str(d) + ' dB')

            plt.legend(loc='upper right', fontsize=12, frameon=True)
            sns.despine()
            plt.grid(False)
            plt.xlabel('Frequency (kHz)', size=14)
            if self.ui.radioButtonMeanSpikes.isChecked():
                plt.ylabel('Mean Spikes Per Presentation', size=14)
            elif self.ui.radioButtonResponseRate.isChecked():
                plt.ylabel('Response Rate (Hz)', size=14)
            elif self.ui.radioButtonOther.isChecked():
                plt.ylabel('Other ???', size=14)

            plt.tick_params(axis='both', which='major', labelsize=14)

            plt.title(str.split(str(filename), '/')[-1].replace('.hdf5', '') + ' '
                      + str(self.ui.comboBox_test_num.currentText()).replace('test_', 'Test '))

            plt.figtext(.02, .02, 'Threshold: ' + str(self.ui.doubleSpinBox_threshold.value()) + ' V')

            plt.show()

            title = str.split(str(filename), '/')[-1].replace('.hdf5', '') + '_' \
                        + str(self.ui.comboBox_test_num.currentText())

            # plt.savefig('output' + os.sep + 'tuning_curves' + os.sep + title + '_tuning_curve_1.png')

            check_output_folders()

            try:
                out_file = open('output' + os.sep + 'tuning_curves' + os.sep + title + '_tuning_curve_1.csv', 'wb')
            except IOError, e:
                self.add_message('Unable to open ' + str(filename) + '\nError ' + str(e.errno) + ': ' + e.strerror + '\n')
            writer = csv.writer(out_file)

            writer.writerow(['File:', filename])
            writer.writerow(['Test:', self.ui.comboBox_test_num.currentText()])
            writer.writerow(['Threshold (V):', self.ui.doubleSpinBox_threshold.value()])
            writer.writerow([])

            for d in db:
                tc_intensity = list(tuningCurves[tuningCurves['intensity'] == d]['intensity'])[0]
                tc_freq = list(tuningCurves[tuningCurves['intensity'] == d]['freq'])
                tc_response = list(tuningCurves[tuningCurves['intensity'] == d]['response'])

                writer.writerow(['Intensity (dB):', tc_intensity])
                writer.writerow(['Frequency (kHz):'] + tc_freq)
                writer.writerow(['Response (Hz):'] + tc_response)
                writer.writerow([])

            out_file.close()
        elif self.ui.radioButtonContour.isChecked():
            # Black and White Tuning Curve
            tc_plot = plt.figure()

            colorRange = (-10, 10.1)
            I = np.unique(np.array(tuningCurves['intensity']))
            F = np.array(tuningCurves['freq'])
            R = np.array(np.zeros((len(I), len(F))))
            for ci, i in enumerate(I):
                for cf, f in enumerate(F):
                    R[ci, cf] = tuningCurves['response'].where(tuningCurves['intensity'] == i).where(
                        tuningCurves['freq'] == f).dropna().values[0]
            levelRange = np.arange(colorRange[0], colorRange[1],
                                   (colorRange[1] - colorRange[0]) / float(25 * (colorRange[1] - colorRange[0])))
            sns.set_context(rc={"figure.figsize": (7, 4)})
            ax = plt.contourf(F, I, R)  # , vmin=colorRange[0], vmax=colorRange[1], levels=levelRange, cmap = cm.bwr )
            plt.colorbar()
            # plt.title(unit, fontsize=14)
            plt.xlabel('Frequency (kHz)', fontsize=14)
            plt.ylabel('Intensity (dB)', fontsize=14)

            plt.title(str.split(str(filename), '/')[-1].replace('.hdf5', '') + ' '
                      + str(self.ui.comboBox_test_num.currentText()).replace('test_', 'Test '))

            plt.figtext(.02, .02, 'Threshold: ' + str(self.ui.doubleSpinBox_threshold.value()) + ' V')

            tc_plot.show()

    def generate_rasters(self):
        filename = self.filename

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

    def auto_threshold(self):
        thresh_fraction = 0.7

        filename = self.filename

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
            trace_data = trace_data[:, :, 1, :]
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

    def update_thresh(self):
        self.ui.view.setThreshold(self.ui.doubleSpinBox_threshold.value())
        self.ui.view.update_thresh()

    def update_thresh2(self):
        self.ui.doubleSpinBox_threshold.setValue(self.ui.view.getThreshold())

    def valid_filename(self, filename):
        # Validate filename
        if filename != '':
            if '.hdf5' in filename:
                try:
                    temp_file = h5py.File(unicode(self.filename), 'r')
                    temp_file.close()
                except IOError:
                    return False
            else:
                return False
        else:
            return False

        return True

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
