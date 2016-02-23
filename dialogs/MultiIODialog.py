import h5py
import operator
import numpy as np
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt

from PyQt4 import QtCore, QtGui
from ui.multi_io_ui import Ui_Form_multi_io

from util.spikerates import ResponseStats

from util.spikestats import get_spike_times


class MultiIODialog(QtGui.QMainWindow):
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