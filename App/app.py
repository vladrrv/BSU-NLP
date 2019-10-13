import os
import numpy as np
import nltk
from nltk.corpus.reader.plaintext import *
import sys
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from Corpus import *

qtCreatorFile = "app_window.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class CorpusLoadTask(QThread):
    done = pyqtSignal()
    busy_sig = pyqtSignal(bool)

    def __init__(self, corpus_dir):
        super(CorpusLoadTask, self).__init__()
        self.corpus_dir = corpus_dir
        self.corpus = Corpus()

    def run(self):
        self.busy_sig.emit(True)
        corpus_reader = PlaintextCorpusReader(self.corpus_dir, '.*')
        raw_text = corpus_reader.raw()
        self.corpus.add_text(raw_text)

        self.busy_sig.emit(False)
        self.done.emit()


class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.actionAdd_Corpus_Directory.triggered.connect(self.open_dir)
        self.actionSave_Dictionary.triggered.connect(self.save_corpus)
        self.actionLoad_Dictionary.triggered.connect(self.load_corpus)
        self.searchButton.clicked.connect(self.search)
        self.editButton.clicked.connect(self.edit_word)
        self.lineEdit.returnPressed.connect(self.search)
        self.lineEdit_2.returnPressed.connect(self.edit_word)
        self.tableWidget.itemActivated.connect(self.on_item_select)
        self.corpus = Corpus()

    def on_corpus_loaded(self):
        self.corpus = self.corpus_load_task.corpus
        self.load_words(self.corpus.get_words())

    def switch_progress_range(self, is_busy):
        if is_busy:
            self.progressBar.setRange(0, 0)
            print('busy')
        else:
            self.progressBar.setRange(0, 100)
            print('free')

    def init_corpus_load_task(self, corpus_dir):
        self.corpus_load_task = CorpusLoadTask(corpus_dir)
        self.corpus_load_task.done.connect(self.on_corpus_loaded)
        self.corpus_load_task.busy_sig.connect(self.switch_progress_range)
        self.corpus_load_task.start()

    def open_dir(self):
        corpus_dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        if corpus_dir is None or corpus_dir == '':
            return
        self.init_corpus_load_task(corpus_dir)

    def search(self):
        char_seq = self.lineEdit.text()
        print(char_seq)
        found = self.corpus.get_words()
        if not (char_seq is None or char_seq == ''):
            reg = "^"+char_seq
            found = [word for word in found if re.match(reg, word) is not None]
        self.load_words(found)

    def on_item_select(self, item):
        if item.column() == 0:
            word = item.text()
            print(word)
            context = self.corpus.find_word_context(word)
            self.textBrowser.setText(context)
            self.lineEdit_2.setText(word)
            self.lineEdit_2.setReadOnly(False)
            self.ed_word = word

    def edit_word(self):
        new = self.lineEdit_2.text()
        self.corpus.replace_word(self.ed_word, new)
        self.load_words(self.corpus.get_words())
        self.textBrowser.setText('')
        self.lineEdit_2.setText('')
        self.lineEdit_2.setReadOnly(True)

    def load_words(self, word_list):
        while self.tableWidget.rowCount() > 0:
            self.tableWidget.removeRow(0)
        self.tableWidget.setSortingEnabled(False)
        for word in word_list:
            current_row_count = self.tableWidget.rowCount()
            self.tableWidget.insertRow(current_row_count)
            item = QTableWidgetItem(word)
            self.tableWidget.setItem(current_row_count, 0, item)
            item = QTableWidgetItem()
            item.setData(Qt.DisplayRole, self.corpus.get_freq(word))
            self.tableWidget.setItem(current_row_count, 1, item)
            item = QTableWidgetItem()
            item.setData(Qt.DisplayRole, self.corpus.get_tag(word))
            self.tableWidget.setItem(current_row_count, 2, item)
        self.tableWidget.setSortingEnabled(True)

    def save_corpus(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Corpus")
        if filename is None or filename == '':
            return
        with open(filename, 'wb') as handle:
            pickle.dump(self.corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_corpus(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Corpus")
        if filename is None or filename == '':
            return
        with open(filename, 'rb') as handle:
            self.corpus = pickle.load(handle)
        self.load_words(self.corpus.get_words())


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MyApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(e)
