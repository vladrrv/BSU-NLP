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
        files = corpus_reader.fileids()
        try:
            for file in files:
                with open(os.path.join(self.corpus_dir, file), 'r', encoding='utf-8') as f:
                    text = f.read()
                self.corpus.add_text(text, file)
        except Exception as e:
            print(e)
        self.busy_sig.emit(False)
        self.done.emit()


class MyApp(QMainWindow, Ui_MainWindow):

    cur_num = 0
    cur_word = None
    cur_tag = None
    cur_text_name = None
    cur_index = None
    cur_word_annot = None
    cur_tag_annot = None

    def _init_fields(self):
        self.pb_query: QPushButton = self.pb_query

        self.tabs: QTabWidget = self.tabs

        self.action_add: QAction = self.action_add
        self.action_save: QAction = self.action_save
        self.action_load: QAction = self.action_load
        self.action_td: QAction = self.action_td
        self.action_collect: QAction = self.action_collect
        self.action_annotate: QAction = self.action_annotate

        self.progress_bar: QProgressBar = self.progress_bar
        self.label_progress: QLabel = self.label_progress

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

        self.tb_raw: QTextBrowser = self.tb_raw
        self.lw_raw: QListWidget = self.lw_raw

        self.tab_annotated: QWidget = self.tab_annotated
        self.te_annotated: QTextEdit = self.te_annotated
        self.le_annotated: QLineEdit = self.le_annotated
        self.cb_annotated: QComboBox = self.cb_annotated
        self.grid_legend: QGridLayout = self.grid_legend
        self.pb_edit_annot: QPushButton = self.pb_edit_annot

        self.tw_stat_t: QTableWidget = self.tw_stat_t
        self.tw_stat_wt: QTableWidget = self.tw_stat_wt
        self.tw_stat_tt: QTableWidget = self.tw_stat_tt

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
            label = QLineEdit()
            label.setText(', '.join(tags))
            label.setReadOnly(True)
            label.setFixedSize(80, 20)
            label.setStyleSheet(f"border: none;")
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

        self.pb_query.clicked.connect(self.query)

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

        self.lw_raw.selectionModel().selectionChanged.connect(self.on_raw_select)

        self.te_annotated.viewport().installEventFilter(self)
        self.cb_annotated.addItems(list(tag_colormap.keys()))
        self.pb_edit_annot.clicked.connect(self.edit_annot)

        self.corpus = Corpus()

    def eventFilter(self, obj, event):
        if obj == self.te_annotated.viewport():
            if event.type() == QEvent.MouseButtonRelease:
                self.peek_word()
                return True
        return False

    def set_status(self, status):
        self.label_progress.setText(status)

    def switch_progress_range(self, is_busy):
        if is_busy:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)

    def run_corpus_load_task(self, corpus_dir):
        corpus_load_task = CorpusLoadTask(corpus_dir, self.corpus)
        self.corpus.status_sig.connect(self.set_status)

        def on_corpus_loaded():
            self.corpus = corpus_load_task.corpus
            words, modified = self.corpus.get_words()
            self.load_words(words, modified)
            self.lw_raw.clear()
            self.lw_raw.addItems(self.corpus.get_text_names())
            self.set_status('Ready')

        corpus_load_task.done.connect(on_corpus_loaded)
        corpus_load_task.busy_sig.connect(self.switch_progress_range)
        corpus_load_task.start()

    def open_dir(self):
        corpus_dir = QFileDialog.getExistingDirectory(self, "Select Directory", "../")
        if corpus_dir is None or corpus_dir == '':
            return
        self.run_corpus_load_task(corpus_dir)

    def show_td(self):
        TagsDescriptionDialog(self).exec_()

    def search(self):
        char_seq = self.le_search.text()
        found, modified = self.corpus.get_words()
        if not (char_seq is None or char_seq == ''):
            reg = "^"+char_seq
            found = [word for word in found if re.match(reg, word) is not None]
        self.load_words(found, modified)

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
            index = self.find_index(self.cur_word, self.cur_num)
            self.corpus.replace_word(index, new)
            words, modified = self.corpus.get_words()
            self.load_words( words, modified)
            self.tb_context.setText('')
            self.le_editword.setText('')
            self.le_editword.setReadOnly(True)
            self.lw_tags.clear()
            self.gb_word.setEnabled(False)
            self.pb_removetag.setEnabled(False)
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
        tok_index, word, tag = self.corpus.find_word_by_raw_index(self.cur_text_name, i)
        self.cur_index = tok_index
        if word is not None:
            self.cur_word_annot = word
            self.cur_tag_annot = tag
            self.le_annotated.setText(word)
            self.cb_annotated.setCurrentText(tag)
            self.cb_annotated.setEnabled(True)
            self.le_annotated.setEnabled(True)
            self.pb_edit_annot.setEnabled(True)
            start, end = self.corpus.get_word_bounds(self.cur_text_name, tok_index)
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.KeepAnchor)
            self.te_annotated.setTextCursor(cursor)

    def edit_annot(self):
        cursor = self.te_annotated.textCursor()
        word = self.le_annotated.text()
        tag = self.cb_annotated.currentText()
        if word != self.cur_word_annot:
            self.corpus.replace_word(self.cur_index, word)
            self.load_words(*self.corpus.get_words())
            cursor.insertText(word)
        elif tag != self.cur_tag_annot:
            self.corpus.replace_tag(self.cur_index, tag)
            format = cursor.charFormat()
            format.setBackground(QColor(tag_colormap[tag]))
            cursor.setCharFormat(format)

    def load_annotated(self):
        self.cur_word_annot = None
        self.cur_tag_annot = None
        annotated_text = self.corpus.get_annotated_text(self.cur_text_name)
        self.te_annotated.clear()
        self.te_annotated.append(annotated_text)
        self.tabs.setCurrentIndex(2)

    def on_raw_select(self):
        selection = self.lw_raw.selectedItems()
        if len(selection) == 0:
            self.cur_text_name = None
            self.tb_raw.clear()
            self.tb_raw.setEnabled(False)
            self.action_annotate.setEnabled(False)
            return

        self.cur_text_name = selection[0].text()
        raw_text = self.corpus.get_raw_text(self.cur_text_name)
        self.tb_raw.setEnabled(True)
        self.tb_raw.setText(raw_text)
        self.action_annotate.setEnabled(True)

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

    def load_words(self, word_list, modified=set()):
        self.tw_wordfreq.setSortingEnabled(False)
        self.tw_wordfreq.setRowCount(0)
        self.tw_wordfreq.setHorizontalHeaderLabels(['Word', 'Frequency'])
        for word in word_list:
            current_row_count = self.tw_wordfreq.rowCount()
            self.tw_wordfreq.insertRow(current_row_count)
            item = QTableWidgetItem(word)
            if word in modified:
                item.setBackground(QColor('#ddffdd'))
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
        self.corpus.save_to_pickle(filename)

    def load_corpus(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Corpus", "../", "Pickle files (*.pkl)")
        if filename is None or filename == '':
            return
        self.corpus.load_from_pickle(filename)
        words, modified = self.corpus.get_words()
        self.load_words(words, modified)
        self.lw_raw.clear()
        self.lw_raw.addItems(self.corpus.get_text_names())

    def query(self):
        relevant_texts = self.corpus.get_text_names()
        try:
            phrase = self.le_query.text()
            if phrase:
                relevant_texts, _ = self.corpus.query(phrase)
            self.lw_raw.clear()
            self.lw_raw.addItems(list(relevant_texts))
        except Exception as e:
            print(e)


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MyApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(e)
