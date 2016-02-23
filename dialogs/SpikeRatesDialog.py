import operator

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt4 import QtCore, QtGui

from ui.spike_rates_ui import Ui_Form_spike_rates
from util import spikestats


class SpikeRatesDialog(QtGui.QMainWindow):
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