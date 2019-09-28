import os
import re
import numpy as np
import nltk
from nltk.corpus.reader.plaintext import *
from nltk.tokenize import *
from nltk import FreqDist
import sys
import time
from PyQt5 import QtGui, uic, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

qtCreatorFile = "app_window.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class CorpusLoadTask(QThread):
    done = pyqtSignal()
    progress_sig = pyqtSignal(int)
    busy_sig = pyqtSignal(bool)

    def __init__(self, corpus_dir):
        super(CorpusLoadTask, self).__init__()
        self.corpus_dir = corpus_dir
        self.corpus = None
        self.raw_text = ''
        self.words = []
        self.words_ids = []
        self.unique_words = []
        self.freq_dict = {}

    def run(self):
        self.busy_sig.emit(True)
        self.corpus = PlaintextCorpusReader(self.corpus_dir, '.*')
        self.raw_text = self.corpus.raw().replace("''", ',,').lower()
        tokenizer = TreebankWordTokenizer()
        gen = tokenizer.span_tokenize(self.raw_text)

        raw_text_len = len(self.raw_text)
        spans = []
        span = next(gen, None)
        self.busy_sig.emit(False)
        while span is not None:
            spans.append(span)
            progress = (span[1] + 1) / raw_text_len * 100
            if int(progress*1000) % 1000 == 0:
                self.progress_sig.emit(progress)
            span = next(gen, None)

        self.progress_sig.emit(100)
        self.busy_sig.emit(True)
        reg = "[A-Za-z]+([\'|\-][A-Za-z]+)*"
        self.words_ids = [(self.raw_text[s:t], s) for s, t in spans if re.fullmatch(reg, self.raw_text[s:t]) is not None]
        self.words = [w for w, i in self.words_ids]
        self.freq_dict = FreqDist(self.words)
        self.unique_words = list(self.freq_dict.keys())
        self.unique_words.sort()
        self.busy_sig.emit(False)
        self.done.emit()


class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.actionAdd_Corpus_Directory.triggered.connect(self.open_dir)
        self.searchButton.clicked.connect(self.search)
        self.lineEdit.returnPressed.connect(self.search)
        self.tableWidget.itemActivated.connect(self.on_item_select)
        self.raw_text = ''
        self.words = []
        self.words_ids = []
        self.unique_words = []
        self.freq_dict = {}

    def on_corpus_loaded(self):
        self.freq_dict = self.corpus_load_task.freq_dict
        self.words = self.corpus_load_task.words
        self.words_ids = self.corpus_load_task.words_ids
        self.unique_words = self.corpus_load_task.unique_words
        self.raw_text = self.corpus_load_task.raw_text
        print('Total:', len(self.words_ids))
        self.load_words(self.unique_words)

    def switch_progress_range(self, is_busy):
        if is_busy:
            self.progressBar.setRange(0, 0)
            print('busy')
        else:
            self.progressBar.setRange(0, 100)
            print('free')

    def open_dir(self):
        corpus_dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        if corpus_dir is None or corpus_dir == '':
            return
        self.corpus_load_task = CorpusLoadTask(corpus_dir)
        self.corpus_load_task.done.connect(self.on_corpus_loaded)
        self.corpus_load_task.progress_sig.connect(lambda p: self.progressBar.setValue(p))
        self.corpus_load_task.busy_sig.connect(self.switch_progress_range)
        self.corpus_load_task.start()

    def search(self):
        char_seq = self.lineEdit.text()
        print(char_seq)
        found = self.unique_words
        if not (char_seq is None or char_seq == ''):
            reg = "^"+char_seq
            found = [word for word in self.unique_words if re.match(reg, word) is not None]
        self.load_words(found)

    def on_item_select(self, item):
        word = item.text()
        print(word)
        context = self.find_word_context(word)
        self.textBrowser.setText(context)

    def find_word_context(self, word):
        i = self.raw_text.find(word)
        context = self.raw_text[max(0, i-100):i+100]
        return "..."+context+"..."

    def load_words(self, word_list):
        while self.tableWidget.rowCount() > 0:
            self.tableWidget.removeRow(0)
        for word in word_list:
            current_row_count = self.tableWidget.rowCount()
            self.tableWidget.insertRow(current_row_count)
            item = QTableWidgetItem(word)

            self.tableWidget.setItem(current_row_count, 0, item)
            item = QTableWidgetItem()
            item.setData(Qt.DisplayRole, self.freq_dict[word])
            self.tableWidget.setItem(current_row_count, 1, item)
        self.tableWidget.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
