import h5py
import operator
import numpy as np

from operator import attrgetter
from collections import namedtuple

from PyQt4 import QtCore, QtGui
from ui.abr_ui import Ui_Form_abr


class ABRDialog(QtGui.QMainWindow):
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
                    try:
                        frequency.append(stim_info[t]['components'][0]['frequency'])
                    except:
                        # print stim_info[t]['components'][0]
                        frequency.append(0)  # TODO: Find actual frequency
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
                    try:
                        frequency.append(stim_info[t]['components'][0]['frequency'])
                    except:
                        # print stim_info[t]['components'][0]
                        frequency.append(0)  # TODO: Find actual frequency
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

        try:
            comment = h_file[target_seg].attrs['comment']
            self.ui.lineEdit_comments.setText(comment)
        except:
            print 'Failed to load comment'

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
                    try:
                        frequency.append(stim_info[t]['components'][0]['frequency'])
                    except:
                        # print stim_info[t]['components'][0]
                        frequency.append(0)  # TODO: Find actual frequency
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
                    try:
                        frequency.append(stim_info[t]['components'][0]['frequency'])
                    except:
                        # print stim_info[t]['components'][0]
                        frequency.append(0)  # TODO: Find actual frequency
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
                ymin = 100
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
