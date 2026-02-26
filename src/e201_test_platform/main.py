import sys
import time
from dataclasses import dataclass
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui

from e201_test_platform.gui.ui_template import Ui_MainWindow


class E201TestPlatform(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

def main():
    app = QtWidgets.QApplication(sys.argv)

    # Optional: nicer fonts on Windows
    app.setFont(QtGui.QFont("Segoe UI", 9))

    w = E201TestPlatform()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()