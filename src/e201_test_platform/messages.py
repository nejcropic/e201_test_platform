from PyQt5.QtWidgets import QMessageBox


from PyQt5.QtCore import Qt


def show_question(text, info="", details=None):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)

    # Allow HTML
    msg.setTextFormat(Qt.RichText)
    msg.setText(text)
    if info:
        msg.setInformativeText(f"<i>{info}</i>")

    if details:
        msg.setDetailedText(details)

    msg.setWindowTitle("Message")
    msg.setMinimumWidth(500)

    msg.setStyleSheet("""
        QMessageBox {
            background-color: #fdfdfd;
        }
        QLabel {
            min-width: 400px;
            qproperty-alignment: AlignLeft | AlignVCenter;
            font-size: 11pt;
        }
        QPushButton {
            padding: 6px 16px;
            border-radius: 6px;
        }
    """)

    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    result = msg.exec_()
    return result != QMessageBox.Cancel


def show_notification(text):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(text)
    msg.setInformativeText("")
    msg.setWindowTitle("Notification")
    msg.setDetailedText("")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()


def show_warning(text, error=""):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText(text)
    msg.setInformativeText(error)
    msg.setWindowTitle("Warning")
    msg.setDetailedText("")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()


def show_error(text, error):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText(text)
    msg.setInformativeText("")
    msg.setWindowTitle("Error")
    msg.setDetailedText(error)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.setMinimumWidth(300)  # Wider box
    msg.setStyleSheet("QLabel{min-width: 250;}")
    msg.exec_()
