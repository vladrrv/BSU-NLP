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


class Corpus:
    def __init__(self, raw_text, freq_dict):
        self.raw_text = raw_text
        self.freq_dict = freq_dict

    def merge(self, corpus):
        new_dict = self.freq_dict.copy()
        for key in list(corpus.freq_dict.keys()):
            new_dict[key] = new_dict.get(key, 0) + corpus.freq_dict[key]

        new_text = self.raw_text + '\n\n**\n\n' + corpus.raw_text

        return Corpus(new_text, new_dict)

    def get_words(self):
        return list(self.freq_dict.keys())

    def find_word_context(self, word):
        i = self.raw_text.find(word)
        context = self.raw_text[max(0, i-100):i+100]
        return "..."+context+"..."

    def replace_word(self, old, new):
        l = len(old)
        print(len(self.raw_text))
        starts = [m.start() for m in re.finditer('(^|[^\w\-\'])('+old+')([^\w\-\']|$)', self.raw_text)]
        new_text = ''
        prev_end = 0
        for start in starts:
            new_text += self.raw_text[prev_end:start] + new
            prev_end = start+l
        new_text += self.raw_text[prev_end:]
        self.raw_text = new_text

        cnt = self.freq_dict.pop(old)
        new_words = new.split()
        for w in new_words:
            self.freq_dict[w] = self.freq_dict.get(w, 0) + cnt


class CorpusLoadTask(QThread):
    done = pyqtSignal()
    busy_sig = pyqtSignal(bool)

    def __init__(self, corpus_dir):
        super(CorpusLoadTask, self).__init__()
        self.corpus_dir = corpus_dir
        self.corpus = None
        self.raw_text = ''
        self.freq_dict = {}

    def get_corpus(self):
        return Corpus(self.raw_text, self.freq_dict)

    def run(self):
        self.busy_sig.emit(True)
        self.corpus = PlaintextCorpusReader(self.corpus_dir, '.*')
        self.raw_text = self.corpus.raw().replace("''", '"').lower()
        tokenizer = TreebankWordTokenizer()
        spans = list(tokenizer.span_tokenize(self.raw_text))

        reg = "[A-Za-z]+([\'|\-][A-Za-z]+)*"
        words_ids = [(self.raw_text[s:t], s) for s, t in spans if re.fullmatch(reg, self.raw_text[s:t]) is not None]
        words = [word_id[0] for word_id in words_ids]
        self.freq_dict = FreqDist(words)
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
        self.lineEdit.returnPressed.connect(self.search)
        self.lineEdit_2.returnPressed.connect(self.edit_word)
        self.tableWidget.itemActivated.connect(self.on_item_select)
        self.corpus = Corpus('', {})

    def on_corpus_loaded(self):
        self.corpus = self.corpus.merge(self.corpus_load_task.get_corpus())
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
            item.setData(Qt.DisplayRole, self.corpus.freq_dict[word])
            self.tableWidget.setItem(current_row_count, 1, item)
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
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
