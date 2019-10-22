import os
import numpy as np
import nltk
from nltk.corpus.reader.plaintext import *
import sys
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from Corpus import *
from td import TagsDescriptionDialog

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
        self.action_td.triggered.connect(self.show_td)
        self.searchButton.clicked.connect(self.search)
        self.editButton.clicked.connect(self.edit_word)
        self.le_search.returnPressed.connect(self.search)
        self.le_editword.returnPressed.connect(self.edit_word)
        self.tableWidget.itemActivated.connect(self.on_word_select)

        self.lw_tags.itemActivated.connect(self.on_tag_select)
        self.pb_addtag.clicked.connect(self.add_tag)
        self.pb_removetag.clicked.connect(self.remove_tag)

        self.cb_tags.addItems(POS_TAGS)

        self.corpus = Corpus()
        self.ed_word = None
        self.ed_tag = None

    def switch_progress_range(self, is_busy):
        if is_busy:
            self.progressBar.setRange(0, 0)
            print('busy')
        else:
            self.progressBar.setRange(0, 100)
            print('free')

    def init_corpus_load_task(self, corpus_dir):
        corpus_load_task = CorpusLoadTask(corpus_dir)

        def on_corpus_loaded():
            self.corpus = corpus_load_task.corpus
            self.load_words(self.corpus.get_words())

        corpus_load_task.done.connect(on_corpus_loaded)
        corpus_load_task.busy_sig.connect(self.switch_progress_range)
        corpus_load_task.start()

    def open_dir(self):
        corpus_dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        if corpus_dir is None or corpus_dir == '':
            return
        self.init_corpus_load_task(corpus_dir)

    def show_td(self):
        TagsDescriptionDialog(self).exec_()

    def search(self):
        char_seq = self.le_search.text()
        print(char_seq)
        found = self.corpus.get_words()
        if not (char_seq is None or char_seq == ''):
            reg = "^"+char_seq
            found = [word for word in found if re.match(reg, word) is not None]
        self.load_words(found)

    def on_word_select(self, item):
        if item.column() == 0:
            word = item.text()
            print(word)
            context = self.corpus.find_word_context(word)
            self.tb_context.setText(context)
            self.le_editword.setText(word)
            self.le_editword.setReadOnly(False)
            self.ed_word = word
            tags = self.corpus.get_tags(word)
            self.load_tags(tags)

    def on_tag_select(self, item):
        self.ed_tag = item.text()
        self.le_initform.setText(self.corpus.get_init_form(self.ed_word, self.ed_tag))
        print(self.ed_tag)

    def edit_word(self):
        new = self.le_editword.text()
        self.corpus.replace_word(self.ed_word, new)
        self.load_words(self.corpus.get_words())
        self.tb_context.setText('')
        self.le_editword.setText('')
        self.le_editword.setReadOnly(True)

    def add_tag(self):
        tag = self.cb_tags.currentText()
        if tag in POS_TAGS:
            word = self.ed_word
            self.corpus.add_tag(word, tag)
            self.load_tags(self.corpus.get_tags(word))

    def remove_tag(self):
        tag = self.ed_tag
        self.ed_tag = ''
        if tag is not None and tag != '':
            word = self.ed_word
            self.corpus.remove_tag(word, tag)
            self.load_tags(self.corpus.get_tags(word))

    def load_words(self, word_list):
        self.tableWidget.setSortingEnabled(False)
        self.tableWidget.clear()
        self.tableWidget.setHorizontalHeaderLabels(['Word','Frequency'])
        for word in word_list:
            current_row_count = self.tableWidget.rowCount()
            self.tableWidget.insertRow(current_row_count)
            item = QTableWidgetItem(word)
            self.tableWidget.setItem(current_row_count, 0, item)
            item = QTableWidgetItem()
            item.setData(Qt.DisplayRole, self.corpus.get_freq(word))
            self.tableWidget.setItem(current_row_count, 1, item)
        self.tableWidget.setSortingEnabled(True)

    def load_tags(self, tags):
        self.lw_tags.clear()
        self.lw_tags.addItems(list(tags))

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
