import os
import numpy as np
import nltk
from nltk.corpus.reader.plaintext import *
import sys
import math
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from Corpus import *
from td import TagsDescriptionDialog

qtCreatorFile = "app_window.ui"

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class CorpusLoadTask(QThread):
    done = pyqtSignal()
    busy_sig = pyqtSignal(bool)

    def __init__(self, corpus_dir, corpus):
        super(CorpusLoadTask, self).__init__()
        self.corpus_dir = corpus_dir
        self.corpus = corpus

    def run(self):
        self.busy_sig.emit(True)
        corpus_reader = PlaintextCorpusReader(self.corpus_dir, '.*')
        raw_text = corpus_reader.raw()
        self.corpus.add_text(raw_text)

        self.busy_sig.emit(False)
        self.done.emit()


class MyApp(QMainWindow, Ui_MainWindow):

    cur_num = 0
    cur_word = None
    cur_tag = None
    cur_word_annot = None
    cur_tag_annot = None

    def _init_fields(self):
        self.action_add: QAction = self.action_add
        self.action_save: QAction = self.action_save
        self.action_load: QAction = self.action_load
        self.action_td: QAction = self.action_td
        self.action_collect: QAction = self.action_collect
        self.action_annotate: QAction = self.action_annotate

        self.pb_search: QPushButton = self.pb_search
        self.pb_edit: QPushButton = self.pb_edit
        self.le_search: QLineEdit = self.le_search
        self.le_editword: QLineEdit = self.le_editword
        self.tw_wordfreq: QTableWidget = self.tw_wordfreq
        self.tb_context: QTextBrowser = self.tb_context
        self.lw_tags: QListWidget = self.lw_tags
        self.pb_addtag: QPushButton = self.pb_addtag
        self.pb_removetag: QPushButton = self.pb_removetag
        self.cb_tags: QComboBox = self.cb_tags
        self.pb_prev: QPushButton = self.pb_prev
        self.pb_next: QPushButton = self.pb_next

        self.te_annotated: QTextEdit = self.te_annotated
        self.le_annotated: QLineEdit = self.le_annotated
        self.cb_annotated: QComboBox = self.cb_annotated
        self.grid_legend: QGridLayout = self.grid_legend
        self.pb_edit_annot: QPushButton = self.pb_edit_annot

    def set_legend(self):
        colors = {}
        for tag, color in tag_colormap.items():
            if color not in colors:
                colors[color] = [tag]
            else:
                colors[color].append(tag)

        cols = 2
        rows = math.ceil(len(colors.keys())/cols)
        for k, (color, tags) in enumerate(colors.items()):
            i = k // cols
            j = k % cols
            icon = QLabel()
            icon.setStyleSheet(f"background-color: {color};")
            icon.setFixedSize(20, 20)
            label = QLabel(','.join(tags))
            self.grid_legend.addWidget(icon, i, j*2)
            self.grid_legend.addWidget(label, i, j*2+1)

    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.set_legend()

        self.action_add.triggered.connect(self.open_dir)
        self.action_save.triggered.connect(self.save_corpus)
        self.action_load.triggered.connect(self.load_corpus)
        self.action_td.triggered.connect(self.show_td)
        self.action_collect.triggered.connect(self.collect_stats)
        self.action_annotate.triggered.connect(self.load_annotated)

        self.pb_search.clicked.connect(self.search)
        self.pb_edit.clicked.connect(self.edit_word)
        self.le_search.returnPressed.connect(self.search)
        self.le_search.textChanged.connect(self.search)
        self.le_editword.returnPressed.connect(self.edit_word)
        self.tw_wordfreq.itemClicked.connect(self.on_word_select)

        self.lw_tags.itemClicked.connect(self.on_tag_select)
        self.pb_addtag.clicked.connect(self.add_tag)
        self.pb_removetag.clicked.connect(self.remove_tag)
        self.pb_prev.clicked.connect(self.prev_context)
        self.pb_next.clicked.connect(self.next_context)

        self.cb_tags.addItems(POS_TAGS)

        self.te_annotated.viewport().installEventFilter(self)
        self.cb_annotated.currentTextChanged.connect(self.edit_tag)
        self.cb_annotated.addItems(list(tag_colormap.keys()))
        self.pb_edit_annot.clicked.connect(self.edit_annot)

        self.corpus = Corpus()

    def eventFilter(self, object, event):
        if object == self.te_annotated.viewport():
            if event.type() == QEvent.MouseButtonRelease:
                self.peek_word()
                return True
        elif object == self.le_search:
            if event.type() == QEvent.KeyRelease:
                self.search()
                return True
        return False

    def switch_progress_range(self, is_busy):
        if is_busy:
            self.progressBar.setRange(0, 0)
            print('busy')
        else:
            self.progressBar.setRange(0, 100)
            print('free')

    def run_corpus_load_task(self, corpus_dir):
        corpus_load_task = CorpusLoadTask(corpus_dir, self.corpus)

        def on_corpus_loaded():
            self.corpus = corpus_load_task.corpus
            self.load_words(self.corpus.get_words())
            self.load_raw()

        corpus_load_task.done.connect(on_corpus_loaded)
        corpus_load_task.busy_sig.connect(self.switch_progress_range)
        corpus_load_task.start()

    def open_dir(self):
        corpus_dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        if corpus_dir is None or corpus_dir == '':
            return
        self.run_corpus_load_task(corpus_dir)

    def show_td(self):
        TagsDescriptionDialog(self).exec_()

    def search(self):
        char_seq = self.le_search.text()
        found = self.corpus.get_words()
        if not (char_seq is None or char_seq == ''):
            reg = "^"+char_seq
            found = [word for word in found if re.match(reg, word) is not None]
        self.load_words(found)

    def on_word_select(self, item):
        self.gb_word.setEnabled(True)
        self.le_initform.setText('')
        if item.column() != 0:
            item = self.tw_wordfreq.item(item.row(), 0)
        word = item.text()
        self.cur_word = word
        self.cur_num = 0
        self.load_context()
        self.le_editword.setText(word)
        self.le_editword.setReadOnly(False)
        tags = self.corpus.get_tags(word)
        self.load_tags(tags)

    def load_context(self):
        context = self.corpus.get_word_context(self.cur_word, num=self.cur_num)
        self.tb_context.setText(context)
        next_i = self.corpus.find_index(self.cur_word, self.cur_num+1)
        self.pb_prev.setEnabled(self.cur_num > 0)
        self.pb_next.setEnabled(next_i is not None)

    def next_context(self):
        self.cur_num += 1
        self.load_context()

    def prev_context(self):
        self.cur_num -= 1
        self.load_context()

    def on_tag_select(self, item):
        self.le_initform.setEnabled(True)
        self.label_initform.setEnabled(True)
        self.pb_removetag.setEnabled(True)

        self.cur_tag = item.text()
        self.le_initform.setText(self.corpus.get_init_form(self.cur_word, self.cur_tag))

    def edit_word(self):
        try:
            new = self.le_editword.text()
            self.corpus.replace_word(self.cur_word, new, self.cur_num)
            self.load_words(self.corpus.get_words())
            self.tb_context.setText('')
            self.le_editword.setText('')
            self.le_editword.setReadOnly(True)
            self.lw_tags.clear()
            self.gb_word.setEnabled(False)
            self.pb_removetag.setEnabled(False)
        except Exception as e:
            print(e)

    def edit_tag(self):
        try:
            new = self.cb_annotated.currentText()
            print(new)
        except Exception as e:
            print(e)

    def add_tag(self):
        tag = self.cb_tags.currentText()
        if tag in POS_TAGS:
            word = self.cur_word
            self.corpus.add_tag(word, tag)
            self.le_initform.setText('')
            self.le_initform.setEnabled(False)
            self.label_initform.setEnabled(False)
            self.pb_removetag.setEnabled(False)
            self.load_tags(self.corpus.get_tags(word))

    def remove_tag(self):
        tag = self.cur_tag
        self.cur_tag = ''
        if tag is not None and tag != '':
            word = self.cur_word
            self.corpus.remove_tag(word, tag)
            self.le_initform.setText('')
            self.le_initform.setEnabled(False)
            self.label_initform.setEnabled(False)
            self.pb_removetag.setEnabled(False)
            self.load_tags(self.corpus.get_tags(word))

    def collect_stats(self):
        tag_freq, word_tag_freq, tag_tag_freq = self.corpus.collect_stats()
        try:
            self.load_stats(self.tw_stat_t, tag_freq)
            self.load_stats(self.tw_stat_wt, word_tag_freq)
            self.load_stats(self.tw_stat_tt, tag_tag_freq)
        except Exception as e:
            print(e)

    def peek_word(self):
        self.cur_word_annot = None
        self.cur_tag_annot = None
        self.le_annotated.setText('')
        self.le_annotated.setEnabled(False)
        self.cb_annotated.setEnabled(False)
        self.pb_edit_annot.setEnabled(False)
        cursor = self.te_annotated.textCursor()
        i = cursor.position()
        tok_index, word, tag = self.corpus.find_word_by_raw_index(i)
        if word is not None:
            self.cur_word_annot = word
            self.cur_tag_annot = tag
            self.le_annotated.setText(word)
            self.cb_annotated.setCurrentText(tag)
            self.cb_annotated.setEnabled(True)
            self.le_annotated.setEnabled(True)
            self.pb_edit_annot.setEnabled(True)
            start, _, _ = self.corpus.tokenized_text[tok_index]
            end = start + len(word)
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.KeepAnchor)
            self.te_annotated.setTextCursor(cursor)

    def edit_annot(self):
        pass

    def load_annotated(self):
        self.cur_word_annot = None
        self.cur_tag_annot = None
        annotated_text = self.corpus.get_annotated_text()
        self.te_annotated.setText('')
        self.te_annotated.append(annotated_text)

    def load_raw(self):
        raw_text = self.corpus.raw_text
        self.tb_raw.setText(raw_text)

    def load_stats(self, tw, stats):
        tw.setSortingEnabled(False)
        tw.setRowCount(0)
        for key, value in stats.items():
            current_row_count = tw.rowCount()
            tw.insertRow(current_row_count)
            if tw == self.tw_stat_t:
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, str(key))
                tw.setItem(current_row_count, 0, item)
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, value)
                tw.setItem(current_row_count, 1, item)
            else:
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, key[0])
                tw.setItem(current_row_count, 0, item)
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, key[1])
                tw.setItem(current_row_count, 1, item)
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, value)
                tw.setItem(current_row_count, 2, item)

        tw.setSortingEnabled(True)

    def load_words(self, word_list):
        self.tw_wordfreq.setSortingEnabled(False)
        self.tw_wordfreq.setRowCount(0)
        self.tw_wordfreq.setHorizontalHeaderLabels(['Word', 'Frequency'])
        for word in word_list:
            current_row_count = self.tw_wordfreq.rowCount()
            self.tw_wordfreq.insertRow(current_row_count)
            item = QTableWidgetItem(word)
            self.tw_wordfreq.setItem(current_row_count, 0, item)
            item = QTableWidgetItem()
            item.setData(Qt.DisplayRole, self.corpus.get_freq(word))
            self.tw_wordfreq.setItem(current_row_count, 1, item)
        self.tw_wordfreq.setSortingEnabled(True)

    def load_tags(self, tags):
        self.pb_removetag.setEnabled(False)
        self.lw_tags.clear()
        self.lw_tags.addItems(list(tags))

    def save_corpus(self):
        filename, _ = QFileDialog.getSaveFileName(self, "Save Corpus", "../", "Pickle files (*.pkl)")
        if filename is None or filename == '':
            return
        with open(filename, 'wb') as handle:
            pickle.dump(self.corpus, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_corpus(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Corpus", "../", "Pickle files (*.pkl)")
        if filename is None or filename == '':
            return
        with open(filename, 'rb') as handle:
            self.corpus = pickle.load(handle)
        self.load_words(self.corpus.get_words())
        self.load_raw()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MyApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(e)
