from PyQt5 import QtGui
from importlib import resources


def load_pixmap(resource: str) -> QtGui.QPixmap:

    with resources.as_file(resources.files("e201_gui.gui.ui_template").joinpath(resource)) as path:
        return QtGui.QPixmap(str(path))
