from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from Corpus import tag_descr

qtCreatorFile = "td_dialog.ui"

Ui_Dialog, QtBaseClass = uic.loadUiType(qtCreatorFile)


class TagsDescriptionDialog(QDialog, Ui_Dialog):
    def __init__(self, parent):
        QDialog.__init__(self, parent)
        Ui_Dialog.__init__(self)
        self.setupUi(self)
        self.load_tags()

    def load_tags(self):
        self.tw_tags.setSortingEnabled(False)
        for tag, descr in tag_descr.items():
            current_row_count = self.tw_tags.rowCount()
            self.tw_tags.insertRow(current_row_count)
            item = QTableWidgetItem(tag)
            self.tw_tags.setItem(current_row_count, 0, item)
            item = QTableWidgetItem(descr)
            self.tw_tags.setItem(current_row_count, 1, item)
        self.tw_tags.setSortingEnabled(True)
