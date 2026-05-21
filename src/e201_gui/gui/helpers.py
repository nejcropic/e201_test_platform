from PyQt5 import QtGui
from importlib import resources


def load_pixmap(resource: str) -> QtGui.QPixmap:
    return QtGui.QPixmap(load_resource(resource))


def load_icon(resource: str) -> QtGui.QIcon:
    return QtGui.QIcon(load_resource(resource))


def load_resource(resource: str):
    with resources.as_file(resources.files("e201_gui.gui.ui_template").joinpath(resource)) as path:
        return str(path)
