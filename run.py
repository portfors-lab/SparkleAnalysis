import os
import sys
import csv
import h5py
import operator
import spikestats
import numpy as np
import seaborn as sns
import scipy.stats as stats

import pandas as pd
import matplotlib.pyplot as plt

import matplotlib

from operator import attrgetter
from collections import namedtuple

from PyQt4 import QtCore, QtGui
from ui.abr_ui import Ui_Form_abr
from ui.main_ui import Ui_MainWindow
from ui.multi_io_ui import Ui_Form_multi_io
from ui.spike_rates_ui import Ui_Form_spike_rates

matplotlib.style.use('ggplot')


class SpikeRatesPopup(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        self.ui = Ui_Form_spike_rates()
        self.ui.setupUi(self)

        self.filename = ''

        self.checkboxes = []
        self.comboboxes = []
        self.spnboxes = []

        # TODO Enable when implemented
        self.ui.pushButton_auto_threshold.setEnabled(False)
        self.ui.horizontalSlider.setEnabled(False)
        self.ui.doubleSpinBox.setEnabled(False)

        QtCore.QObject.connect(self.ui.pushButton_spike_rates, QtCore.SIGNAL("clicked()"), self.graph_spike_rates)
        QtCore.QObject.connect(self.ui.pushButton_auto_threshold, QtCore.SIGNAL("clicked()"), self.estimate_thresholds)
        QtCore.QObject.connect(self.ui.horizontalSlider, QtCore.SIGNAL("valueChanged(int)"), self.update_spontaneous)
        QtCore.QObject.connect(self.ui.doubleSpinBox, QtCore.SIGNAL("valueChanged(const QString&)"), self.update_spontaneous2)

    def populate_checkboxes(self, filename):
        self.filename = filename

        self.ui.label_title.setText(str.split(str(filename), '/')[-1])
        h_file = h5py.File(unicode(filename), 'r')

        tests = {}
        for key in h_file.keys():
            if 'segment' in key:
                for test in h_file[key].keys():
                    tests[test] = int(test.replace('test_', ''))

        sorted_tests = sorted(tests.items(), key=operator.itemgetter(1))

        # Create the layout to populate
        layout = QtGui.QGridLayout()

        title_test = QtGui.QLabel('Test')
        title_chan = QtGui.QLabel('Channel')
        title_thresh = QtGui.QLabel('Threshold')

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)

        title_test.setFont(font)
        title_chan.setFont(font)
        title_thresh.setFont(font)

        layout.addWidget(title_test, 0, 0)
        layout.addWidget(title_chan, 0, 1)
        layout.addWidget(title_thresh, 0, 2)

        row_count = 1
        for test in sorted_tests:
            checkbox = QtGui.QCheckBox(test[0])
            combobox = QtGui.QComboBox()
            spnbox = QtGui.QDoubleSpinBox()

            layout.addWidget(checkbox, row_count, 0)
            self.checkboxes.append(checkbox)

            # Find target segment
            for segment in h_file.keys():
                for s_test in h_file[segment].keys():
                    if test[0] == s_test:
                        target_seg = segment
                        target_test = s_test

            if len(h_file[target_seg][target_test].value.shape) > 3:
                channels = h_file[target_seg][target_test].value.shape[2]
            else:
                channels = 1

            if channels == 1:
                combobox.addItem('channel_1')
            else:
                for i in range(channels):
                    combobox.addItem('channel_' + str(i+1))

            if combobox.count() < 2:
                combobox.setEnabled(False)

            layout.addWidget(combobox, row_count, 1)
            self.comboboxes.append(combobox)

            spnbox.setSuffix(' V')
            spnbox.setDecimals(4)
            spnbox.setSingleStep(0.0001)
            spnbox.setMinimum(-100)
            spnbox.setMaximum(100)
            # spnbox.setValue(thresh)

            # TODO Enable when implemented
            spnbox.setEnabled(False)

            layout.addWidget(spnbox, row_count, 2)
            self.spnboxes.append(spnbox)

            row_count += 1

        self.ui.scrollAreaWidgetContents.setLayout(layout)

        # Add values to QSlider
        window_time = h_file[target_seg][target_test].value.shape[-1] / h_file[target_seg].attrs['samplerate_ad']
        self.ui.horizontalSlider.setMaximum(window_time * 1000)
        self.ui.horizontalSlider.setValue(window_time * 500)

        h_file.close()

    def estimate_thresholds(self):
        thresh_fraction = 0.7

        h_file = h5py.File(unicode(self.filename), 'r')

        for row in range(len(self.checkboxes)):

            for segment in h_file.keys():
                for test in h_file[segment].keys():
                    if self.checkboxes[row].text() == test:
                        target_seg = segment
                        target_test = test

            trace_data = h_file[target_seg][target_test].value

            if len(trace_data.shape) == 4:
                trace_data = trace_data.squeeze()

            # Still shape of 4
            if len(trace_data.shape) == 4:

                tchan = int(self.comboboxes[row].currentText().replace('channel_', '')) - 1

                trace_data = trace_data[:, :, tchan, :]
                trace_data = trace_data.squeeze()

            # Compute threshold from average maximum of traces
            max_trace = []
            for n in range(len(trace_data[1, :, 0])):
                max_trace.append(np.max(np.abs(trace_data[1, n, :])))
            average_max = np.array(max_trace).mean()
            thresh = thresh_fraction * average_max

            self.spnboxes[row].setValue(thresh)

    def get_spike_trains(self, h_file, test_num, spontStim=1):

        for segment in h_file.keys():
                for test in h_file[segment].keys():
                    if self.checkboxes[test_num].text() == test:
                        target_seg = segment
                        target_test = test

        fs = h_file[target_seg].attrs['samplerate_ad']
        reps = h_file[target_seg][target_test].attrs['reps']

        startTime = h_file[target_seg][target_test].attrs['start']
        trace_data = h_file[target_seg][target_test].value

        if len(trace_data) == 4:
            trace_data = trace_data.squeeze()
        # Still shape of 4
        if len(trace_data.shape) == 4:
            tchan = int(self.comboboxes[test_num].currentText().replace('channel_', '')) - 1
            trace_data = trace_data[:, :, tchan, :]
            trace_data = trace_data.squeeze()

        thresh = self.spnboxes[test_num].value()

        # ----- AutoThreshold -----
        maxTrace = []
        for n in range(len(trace_data[0, :, 0])):
            maxTrace.append(np.max(np.abs(trace_data[spontStim, n, :])))
        aveMax = np.array(maxTrace).mean()
        #         if max(maxTrace) > 1 * np.std(maxTrace):  # remove an extreme outlyer caused by an electrical glitch
        #             maxTrace.remove(max(maxTrace))
        th = 0.7 * aveMax
        thresh = th

        spikeTrains = pd.DataFrame([])
        nspk = 0

        for n in range(len(trace_data[0, :, 0])):
            spikes = spikestats.spike_times(trace_data[spontStim, n, :], threshold=thresh, fs=fs)
            spikeTimes = 1000 * np.array(spikes)
            spikeTimesS = pd.Series(spikeTimes)
            if spikeTimesS.size > nspk:
                spikeTrains = spikeTrains.reindex(spikeTimesS.index)
                nspk = spikeTimesS.size
            spikeTrains[str(n)] = spikeTimesS

        # print 'thresh:', thresh, '\nspikes\n', spikes

        duration = trace_data.shape[-1] / fs

        return spikeTrains, duration, startTime

    def graph_spike_rates(self):
        print 'Spike Rates'

        h_file = h5py.File(unicode(self.filename), 'r')

        target_rows = []
        for i in range(len(self.checkboxes)):
            if self.checkboxes[i].checkState():
                target_rows.append(i)

        axes = []
        timeStamp = []
        spontAve = []
        spontSTD = []
        responseAve = []
        responseSTD = []
        for row in target_rows:

            for segment in h_file.keys():
                for test in h_file[segment].keys():
                    if self.checkboxes[row].text() == test:
                        target_seg = segment
                        target_test = test

            stim_info = eval(h_file[target_seg][target_test].attrs['stim'])
            start_time = h_file[target_seg][target_test].attrs['start']

            spikeTrains, dur, time = self.get_spike_trains(h_file, row, spontStim=0)
            # print '\n----------\n1 SPIKETRAINS:', row, '\n', spikeTrains, '\n----------'

            # --- SpontaneousStats ---
            spontSpikeCount = []
            for k in spikeTrains.keys():
                spk = spikeTrains[k]
                spontSpikeCount.append(len(spk.dropna()) / float(dur))
                if len(spontSpikeCount) > 0:
                    spontStats = [np.mean(spontSpikeCount), np.std(spontSpikeCount)]
                else:
                    spontStats = [0, 0]

            spikeTrains, dur, time = self.get_spike_trains(h_file, row, spontStim=1)
            # print '\n----------\n2 SPIKETRAINS:', row, '\n', spikeTrains, '\n----------'

            # Assumes all stim are the same for the test
            # print 'stim start:', stim_info[1]['components'][0]['start_s']
            # print 'stim duration:', stim_info[1]['components'][0]['duration']
            stimStart = stim_info[1]['components'][0]['start_s']
            stimDuration = stim_info[1]['components'][0]['duration']

            # --- ResponseStats ---
            dur = stimDuration
            responseSpikeCount = []

            for k in spikeTrains.keys():
                spk = spikeTrains[k]
                responseSpikeCount.append(len(spk[spk < stimStart*1000 + stimDuration*1000 + 10]) / dur)
                if len(responseSpikeCount) > 0:
                    responseStats = [np.mean(responseSpikeCount), np.std(responseSpikeCount)]
                else:
                    responseStats = [0, 0]

            timeStamp.append(time)
            spontAve.append(spontStats[0])
            spontSTD.append(spontStats[1])
            responseAve.append(responseStats[0])
            responseSTD.append(responseStats[1])

        # --- Plot the time dependent change in rates ---
        if target_rows:
            # print 'target_rows:', target_rows
            # print self.ui.timeEdit_on_time.date(), self.ui.timeEdit_on_time.time()
            # print self.ui.timeEdit_off_time.date(), self.ui.timeEdit_off_time.time()
            timeOn = str(self.ui.timeEdit_on_time.time().hour()) + ':' \
                     + str(self.ui.timeEdit_on_time.time().minute()) + ':' \
                     + str(self.ui.timeEdit_on_time.time().second())
            timeOff = str(self.ui.timeEdit_off_time.time().hour()) + ':' \
                      + str(self.ui.timeEdit_off_time.time().minute()) \
                      + ':' + str(self.ui.timeEdit_off_time.time().second())
            rateEffects = pd.DataFrame({'Spontaneous': spontAve, 'spontSTD': spontSTD, 'Response': responseAve,
                                        'responseSTD': responseSTD}, index=pd.to_datetime(timeStamp))

        if len(target_rows) > 1:
            fig = plt.figure()
            rateEffects['Response'].plot(yerr=rateEffects['responseSTD'], capthick=1)
            rateEffects['Spontaneous'].plot(yerr=rateEffects['spontSTD'], capthick=1)
            plt.legend(loc='upper right', fontsize=12, frameon=True)
            sns.despine()
            plt.grid(False)
            plt.xlabel('Time (ms)', size=14)
            plt.ylabel('Rate (Hz)', size=14)
            plt.title(str.split(str(self.filename), '/')[-1].replace('.hdf5', ''), size=14)
            plt.tick_params(axis='both', which='major', labelsize=14)
            ax = plt.gca()
            lineY = ax.get_ylim()[1] - 0.01 * ax.get_ylim()[1]
            if timeOff == timeOn: timeOff = rateEffects.index[-1]
            if timeOn != 0 and (isinstance(timeOn, str) and isinstance(timeOff, str)):
                plt.plot((timeOn, timeOff), (lineY, lineY), 'k-', linewidth=4)
            for i, n in enumerate(target_rows):
                plt.annotate(str(n), (rateEffects.index[i], rateEffects['Spontaneous'][i]))

            plt.show()
            h_file.close()

        elif len(target_rows) > 0:
            return rateEffects
        else:
            return []

    def get_pharma_times(self, data):
        pass

    def update_spontaneous(self):
        # print self.ui.horizontalSlider.value()
        self.ui.doubleSpinBox.setValue(self.ui.horizontalSlider.value())

    def update_spontaneous2(self):
        # print self.ui.doubleSpinBox.value()
        self.ui.horizontalSlider.setValue(self.ui.doubleSpinBox.value())


class MultiIOPopup(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self)
        self.ui = Ui_Form_multi_io()
        self.ui.setupUi(self)

        self.filename = ''

        self.checkboxes = []
        self.comboboxes = []
        self.spnboxes = []
        self.threshold = 0

        QtCore.QObject.connect(self.ui.pushButton_multi_io, QtCore.SIGNAL("clicked()"), self.graph_multi_io_test)
        QtCore.QObject.connect(self.ui.pushButton_auto_threshold, QtCore.SIGNAL("clicked()"), self.estimate_thresholds)

    def populate_checkboxes(self, filename, thresh):
        self.filename = filename

        self.ui.label_title.setText(str.split(str(filename), '/')[-1])
        h_file = h5py.File(unicode(filename), 'r')

        tests = {}
        for key in h_file.keys():
            if 'segment' in key:
                for test in h_file[key].keys():
                    tests[test] = int(test.replace('test_', ''))

        sorted_tests = sorted(tests.items(), key=operator.itemgetter(1))

        # Create the layout to populate
        layout = QtGui.QGridLayout()

        title_test = QtGui.QLabel('Test')
        title_chan = QtGui.QLabel('Channel')
        title_thresh = QtGui.QLabel('Threshold')

        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)

        title_test.setFont(font)
        title_chan.setFont(font)
        title_thresh.setFont(font)

        layout.addWidget(title_test, 0, 0)
        layout.addWidget(title_chan, 0, 1)
        layout.addWidget(title_thresh, 0, 2)

        row_count = 1
        for test in sorted_tests:
            checkbox = QtGui.QCheckBox(test[0])
            combobox = QtGui.QComboBox()
            spnbox = QtGui.QDoubleSpinBox()

            layout.addWidget(checkbox, row_count, 0)
            self.checkboxes.append(checkbox)

            # Find target segment
            for segment in h_file.keys():
                for s_test in h_file[segment].keys():
                    if test[0] == s_test:
                        target_seg = segment
                        target_test = s_test

            if len(h_file[target_seg][target_test].value.shape) > 3:
                channels = h_file[target_seg][target_test].value.shape[2]
            else:
                channels = 1

            if channels == 1:
                combobox.addItem('channel_1')
            else:
                for i in range(channels):
                    combobox.addItem('channel_' + str(i+1))

            if combobox.count() < 2:
                combobox.setEnabled(False)

            layout.addWidget(combobox, row_count, 1)
            self.comboboxes.append(combobox)

            spnbox.setSuffix(' V')
            spnbox.setDecimals(4)
            spnbox.setSingleStep(0.0001)
            spnbox.setMinimum(-100)
            spnbox.setMaximum(100)
            spnbox.setValue(thresh)

            layout.addWidget(spnbox, row_count, 2)
            self.spnboxes.append(spnbox)

            row_count += 1

        self.ui.scrollAreaWidgetContents.setLayout(layout)

        h_file.close()

    def estimate_thresholds(self):
        thresh_fraction = 0.7

        h_file = h5py.File(unicode(self.filename), 'r')

        for row in range(len(self.checkboxes)):

            for segment in h_file.keys():
                for test in h_file[segment].keys():
                    if self.checkboxes[row].text() == test:
                        target_seg = segment
                        target_test = test

            trace_data = h_file[target_seg][target_test].value

            if len(trace_data.shape) == 4:
                trace_data = trace_data.squeeze()

            # Still shape of 4
            if len(trace_data.shape) == 4:

                tchan = int(self.comboboxes[row].currentText().replace('channel_', '')) - 1

                trace_data = trace_data[:, :, tchan, :]
                trace_data = trace_data.squeeze()

            # Compute threshold from average maximum of traces
            max_trace = []
            for n in range(len(trace_data[1, :, 0])):
                max_trace.append(np.max(np.abs(trace_data[1, n, :])))
            average_max = np.array(max_trace).mean()
            thresh = thresh_fraction * average_max

            self.spnboxes[row].setValue(thresh)

    def graph_multi_io_test(self):

        h_file = h5py.File(unicode(self.filename), 'r')

        target_rows = []
        for i in range(len(self.checkboxes)):
            if self.checkboxes[i].checkState():
                target_rows.append(i)

        axes = []
        for row in target_rows:

            for segment in h_file.keys():
                for test in h_file[segment].keys():
                    if self.checkboxes[row].text() == test:
                        target_seg = segment
                        target_test = test

            fs = h_file[target_seg].attrs['samplerate_ad']
            reps = h_file[target_seg][target_test].attrs['reps']
            start_time = h_file[target_seg][target_test].attrs['start']
            trace_data = h_file[target_seg][target_test].value

            stim_info = eval(h_file[target_seg][target_test].attrs['stim'])

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
                        tchan = int(self.comboboxes[row].currentText().replace('channel_', '')) - 1
                        trace = trace_data[tStim][tRep][tchan]
                        pass
                    else:
                        self.add_message('Cannot handle trace_data of shape: ' + str(trace_data.shape))
                        return

                    spike_times = 1000 * np.array(get_spike_times(trace, self.spnboxes[row].value(), fs))
                    spike_times_s = pd.Series(spike_times)

                    if spike_times_s.size >= nspk:
                        spikeTrains = spikeTrains.reindex(spike_times_s.index)
                        nspk = spike_times_s.size
                    spikeTrains[str(tRep)] = spike_times_s
                autoRasters[traceKey] = spikeTrains
            rasters = autoRasters

            tuning = []

            sortedKeys = sorted(rasters.keys())
            for traceKey in sortedKeys:
                spl = int(traceKey.split('_')[-1])
                raster = rasters[traceKey]
                res = ResponseStats(raster)
                tuning.append({'intensity': spl, 'response': res[0], 'responseSTD': res[1]})

            tuningCurves = pd.DataFrame(tuning)

            if axes:
                tuningCurves.plot(x='intensity', y='response', ax=axes, yerr='responseSTD', capthick=1, label=str(target_test) + ' : ' + str(self.spnboxes[row].value()) + ' V')
            else:
                axes = tuningCurves.plot(x='intensity', y='response', yerr='responseSTD', capthick=1, label=str(target_test) + ' : ' + str(self.spnboxes[row].value()) + ' V')

        plt.legend(loc='upper left', fontsize=12, frameon=True)
        sns.despine()
        plt.grid(False)
        plt.xlabel('Intensity (dB)', size=14)
        plt.ylabel('Response Rate (Hz)', size=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.title(str.split(str(self.filename), '/')[-1].replace('.hdf5', '') + ' Multi I/O')

        plt.show()
        h_file.close()


class ABRPopup(QtGui.QMainWindow):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_Form_abr()
        self.ui.setupUi(self)

        self.filename = ''

        QtCore.QObject.connect(self.ui.comboBox_test_num, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.populate_comments)
        QtCore.QObject.connect(self.ui.comboBox_test_num, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.populate_frequency)
        QtCore.QObject.connect(self.ui.comboBox_frequency, QtCore.SIGNAL("currentIndexChanged(const QString&)"), self.graph_abr)

        QtCore.QObject.connect(self.ui.doubleSpinBox_min_sep, QtCore.SIGNAL("valueChanged(const QString&)"), self.graph_abr)

    def populate_boxes(self, filename):
        self.filename = filename

        self.ui.label_title.setText(str.split(str(filename), '/')[-1])
        h_file = h5py.File(unicode(filename), 'r')

        self.ui.comboBox_test_num.clear()

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
                    self.ui.label_min_sep.setEnabled(False)
                    self.ui.doubleSpinBox_min_sep.setEnabled(False)
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

    def populate_frequency(self):
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

        self.ui.comboBox_frequency.clear()

        # Makes the assumption that all of the traces are of the same type
        stim_info = eval(h_file[target_seg][target_test].attrs['stim'])

        trace_data = h_file[target_seg][target_test].value

        samples = trace_data.shape[-1]
        traces = trace_data.shape[0]
        reps = trace_data.shape[1]

        average = np.empty(samples)
        abr = np.empty(shape=(traces, 1, 1, samples))

        intensity = []
        frequency = []

        # Average the rep data into one trace
        if len(trace_data.shape) == 4:
            for t in range(traces):
                for s in range(samples):
                    # print s
                    for r in range(reps):
                        # print r
                        average[s] += trace_data[t, r, 0, s]
                    average[s] /= reps + 1
                    abr[t, 0, 0, s] = average[s]
                intensity.append(stim_info[t]['components'][0]['intensity'])
                if t != 0:
                    frequency.append(stim_info[t]['components'][0]['frequency'])
        elif len(trace_data.shape) == 3:
            for t in range(traces):
                for s in range(samples):
                    # print s
                    for r in range(reps):
                        # print r
                        average[s] += trace_data[t, r, s]
                    average[s] /= reps + 1
                    abr[t, 0, 0, s] = average[s]
                intensity.append(stim_info[t]['components'][0]['intensity'])
                if t != 0:
                    frequency.append(stim_info[t]['components'][0]['frequency'])
        else:
            self.add_message('Cannot handle trace_data of shape: ' + str(trace_data.shape))
            return

        # Remove duplicates
        unique = []
        dup = set()
        for freq in frequency:
            if freq not in dup:
                unique.append(freq)
                dup.add(freq)

        for freq in unique:
            self.ui.comboBox_frequency.addItem(str(freq))

        if self.ui.comboBox_frequency.size() > 0:
            self.ui.comboBox_frequency.setEnabled(True)
            self.ui.label_frequency.setEnabled(True)
            self.ui.label_min_sep.setEnabled(True)
            self.ui.doubleSpinBox_min_sep.setEnabled(True)
        else:
            self.ui.comboBox_frequency.setEnabled(False)
            self.ui.label_frequency.setEnabled(False)
            self.ui.label_min_sep.setEnabled(False)
            self.ui.doubleSpinBox_min_sep.setEnabled(False)

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

        # Set Frequency Field disabled
        self.ui.label_frequency.setEnabled(False)
        self.ui.comboBox_frequency.setEnabled(False)
        self.ui.label_min_sep.setEnabled(False)
        self.ui.doubleSpinBox_min_sep.setEnabled(False)

    def graph_abr(self):
        # Assumes there will only be one channel

        print 'Graphing ABR'
        filename = self.filename

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

        fs = h_file[target_seg].attrs['samplerate_ad']

        trace_data = h_file[target_seg][target_test].value

        if len(trace_data.shape) == 4:
            pass

        samples = trace_data.shape[-1]
        traces = trace_data.shape[0]
        reps = trace_data.shape[1]

        average = np.empty(samples)
        temp = np.empty(shape=(traces, 1, 1, samples))

        intensity = []
        frequency = []

        # Average the rep data into one trace
        if len(trace_data.shape) == 4:
            for t in range(traces):
                for s in range(samples):
                    # print s
                    for r in range(reps):
                        # print r
                        average[s] += trace_data[t, r, 0, s]
                    average[s] /= reps + 1
                    temp[t, 0, 0, s] = average[s]
                intensity.append(stim_info[t]['components'][0]['intensity'])
                if t != 0:
                    frequency.append(stim_info[t]['components'][0]['frequency'])
                else:
                    frequency.append(0)
        elif len(trace_data.shape) == 3:
            for t in range(traces):
                for s in range(samples):
                    # print s
                    for r in range(reps):
                        # print r
                        average[s] += trace_data[t, r, s]
                    average[s] /= reps + 1
                    temp[t, 0, 0, s] = average[s]
                intensity.append(stim_info[t]['components'][0]['intensity'])
                if t != 0:
                    frequency.append(stim_info[t]['components'][0]['frequency'])
                else:
                    frequency.append(0)
        else:
            self.add_message('Cannot handle trace_data of shape: ' + str(trace_data.shape))
            return


        freq = []
        inten = []
        trace_num = []

        count = 0

        # Find the number of traces with x frequency
        for t in range(traces):
            if float(self.ui.comboBox_frequency.currentText()) == frequency[t]:
                count += 1

        abr = np.empty(shape=(count, 1, 1, samples))
        count = 0

        # Select only the desired frequency
        for t in range(traces):
            # print 'int: ' + str(intensity[t]) + ' freq: ' + str(frequency[t])

            if float(self.ui.comboBox_frequency.currentText()) == frequency[t]:
                freq.append(frequency[t])
                inten.append(intensity[t])
                trace_num.append(t)
                abr[count, 0, 0, :] = temp[t, 0, 0, :]
                count += 1

        # print 'Select:'
        # for i in range(len(freq)):
        #     print 'int: ' + str(inten[i]) + ' freq: ' + str(freq[i])

        abrtrace = namedtuple('abrtrace', 'samples trace_num frequency intensity')

        abrtraces = []

        for i in range(count):
            abrtraces.append(abrtrace(abr[i, 0, 0, :], trace_num[i], freq[i], inten[i]))

        abrtraces = sorted(abrtraces, key=attrgetter('intensity'))

        for i in range(len(abrtraces)-1):
            for s in range(samples):
                if (abrtraces[i+1].samples[s] - abrtraces[i].samples[s]) < 0 + self.ui.doubleSpinBox_min_sep.value():
                    diff = (abrtraces[i].samples[s] - abrtraces[i+1].samples[s]) + self.ui.doubleSpinBox_min_sep.value()
                    abrtraces[i+1] = abrtrace(abrtraces[i+1].samples + diff, abrtraces[i+1].trace_num, abrtraces[i+1].frequency, abrtraces[i+1].intensity)

        for i in range(len(abrtraces)):
            abr[i] = abrtraces[i].samples
            inten[i] = abrtraces[i].intensity
            trace_num[i] = abrtraces[i].trace_num

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
            self.ui.view.addTracesABR(xlist, abr[:, 0, :], inten, trace_num)
        else:
            self.ui.view.addTracesABR(xlist, abr[:, 0, 0, :], inten, trace_num)

        # Set window size
        if not self.ui.checkBox_custom_window.checkState():
            if len(presentation) > 0:
                ymin = min(presentation)
                ymax = max(presentation)
            else:
                ymin = 0
                ymax = 0
                if len(abr.shape) == 3:
                    rep_len = abr.shape[1]
                    for i in range(rep_len):
                        if min(abr[0, i, :]) < ymin:
                            ymin = min(abr[0, i, :])
                        if max(abr[0, i, :]) > ymax:
                            ymax = max(abr[0, i, :])
                else:
                    trace_len = abr.shape[0]
                    for i in range(trace_len):
                        if min(abr[i, 0, 0, :]) < ymin:
                            ymin = min(abr[i, 0, 0, :])
                        if max(abr[i, 0, 0, :]) > ymax:
                            ymax = max(abr[i, 0, 0, :])

            self.ui.view.setXRange(0, window, 0)
            self.ui.view.setYRange(ymin, ymax, 0.1)

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
        QtCore.QObject.connect(self.ui.pushButton_tuning_curve_1, QtCore.SIGNAL("clicked()"), self.graph_rainbow_tuning_curve)
        QtCore.QObject.connect(self.ui.pushButton_tuning_curve_2, QtCore.SIGNAL("clicked()"), self.graph_tuning_curve)
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

        comment = h_file[target_seg].attrs['comment']
        self.ui.lineEdit_comments.setText(comment)

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

        if self.ui.comboBox_rep.currentText() == '' or self.ui.comboBox_channel.count() < 2:
            self.ui.label_channel.setEnabled(False)
            self.ui.comboBox_channel.setEnabled(False)
            self.ui.comboBox_channel.clear()
        else:
            self.ui.label_channel.setEnabled(True)
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
        self.dialog = SpikeRatesPopup()

        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if not self.valid_filename(filename):
            return

        self.dialog.populate_checkboxes(filename)
        self.dialog.setWindowIcon(QtGui.QIcon('images/horsey.ico'))
        self.dialog.show()

    def graph_multi_io_test(self):
        self.dialog = MultiIOPopup()

        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if not self.valid_filename(filename):
            return

        self.dialog.populate_checkboxes(filename, self.ui.doubleSpinBox_threshold.value())
        self.dialog.setWindowIcon(QtGui.QIcon('images/horsey.ico'))
        self.dialog.show()

    def graph_abr(self):
        self.dialog = ABRPopup()

        filename = self.filename = self.ui.lineEdit_file_name.text()

        # Validate filename
        if not self.valid_filename(filename):
            return

        self.dialog.populate_boxes(filename)
        self.dialog.setWindowIcon(QtGui.QIcon('images/horsey.ico'))
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

    def graph_rainbow_tuning_curve(self):
        filename = self.filename = self.ui.lineEdit_file_name.text()

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
                res = ResponseStats(raster)
                tuning.append({'intensity': spl, 'freq': freq / 1000, 'response': res[0], 'responseSTD': res[1]})

        tuningCurves = pd.DataFrame(tuning)
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
        plt.ylabel('Response Rate (Hz)', size=14)

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

    def graph_tuning_curve(self):
        filename = self.filename = self.ui.lineEdit_file_name.text()

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
                res = ResponseStats(raster)
                tuning.append({'intensity': spl, 'freq': freq / 1000, 'response': res[0], 'responseSTD': res[1]})

        tuningCurves = pd.DataFrame(tuning)

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
        # --- Histogram of spike times (2 ms bins)---
        sns.set_style("white")
        sns.set_style("ticks")
        histogram_f = plt.figure(figsize=(8, 3))
        axHist = spikeTimes.hist(bins=int(duration / 2), range=(0, duration))  # , figsize=(8,3))
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


def ResponseStats(spikeTrains, stimStart=10, stimDuration=50):
    """ Average spike rate during the stimulus response.
    :param spikeTrains, pandas.DataFrame of spike times for each cycle
    :type spikeTrains: pandas.DataFrame
    :param stimStart: Beginning of stimulus time
    :type stimStart: int
    :param stimDuration: Duration of stimulus response
    :type stimDuration: int
    :returns: responseStats, list: average and standard deviation of the rate during the stimulus response
    """
    dur = 0.001 * stimDuration
    responseSpikeCount = []
    spontSpikeCount = []
    for k in spikeTrains.keys():
        spk = spikeTrains[k]
        responseSpikeCount.append(len(spk[spk < stimStart + stimDuration + 10]) / dur)
        spontSpikeCount.append(len(spk[spk > 100]) / 0.1)
        if len(responseSpikeCount) > 0:
            responseStats = [np.mean(responseSpikeCount), np.std(responseSpikeCount)]
        else:
            responseStats = [0, 0]
        if len(spontSpikeCount) > 0:
            spontStats = [np.mean(spontSpikeCount), np.std(spontSpikeCount)]
        else:
            spontStats = [0, 0]
    return responseStats


def get_spike_times(signal, threshold, fs):
    times = []

    if threshold >= 0:
        over, = np.where(signal > float(threshold))
        segments, = np.where(np.diff(over) > 1)
    else:
        over, = np.where(signal < float(threshold))
        segments, = np.where(np.diff(over) > 1)

    if len(over) > 1:
        if len(segments) == 0:
            segments = [0, len(over) - 1]
        else:
            # add end points to sections for looping
            if segments[0] != 0:
                segments = np.insert(segments, [0], [0])
            else:
                # first point in singleton
                times.append(float(over[0]) / fs)
                if 1 not in segments:
                    # make sure that first point is in there
                    segments[0] = 1
            if segments[-1] != len(over) - 1:
                segments = np.insert(segments, [len(segments)], [len(over) - 1])
            else:
                times.append(float(over[-1]) / fs)

        for iseg in range(1, len(segments)):
            if segments[iseg] - segments[iseg - 1] == 1:
                # only single point over threshold
                idx = over[segments[iseg]]
            else:
                segments[0] = segments[0] - 1
                # find maximum of continuous set over max
                idx = over[segments[iseg - 1] + 1] + np.argmax(
                    signal[over[segments[iseg - 1] + 1]:over[segments[iseg]]])
            times.append(float(idx) / fs)
    elif len(over) == 1:
        times.append(float(over[0]) / fs)

    if len(times) > 0:
        refract = 0.002
        times_refract = []
        times_refract.append(times[0])
        for i in range(1, len(times)):
            if times_refract[-1] + refract <= times[i]:
                times_refract.append(times[i])
        return times_refract
    else:
        return times

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myApp = MyForm()
    myApp.setWindowIcon(QtGui.QIcon('images/horsey.ico'))
    myApp.show()
    sys.exit(app.exec_())
