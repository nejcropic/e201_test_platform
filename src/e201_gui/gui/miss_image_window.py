from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class MisImageWindow(QWidget):
    def __init__(self, mis_image_data, position=None):
        super().__init__()

        self.setWindowTitle(f"MIS Image @ {position}°")

        layout = QVBoxLayout(self)

        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)

        self.plot(mis_image_data, position)

    def plot(self, mis_image_data, position):
        ax = self.ax
        ax.clear()

        values = mis_image_data["mis_image_values"]

        ax.plot(values)
        ax.grid(True)

        max_val = np.max(values)
        if max_val < 800:
            ax.set_ylim(-800, 800)
        else:
            ax.set_ylim(-max_val * 1.2, max_val * 1.2)

        text = (
            f"Absolute position code 1: {mis_image_data['abs_pos_code_1']}\n"
            f"Absolute position code 2: {mis_image_data['abs_pos_code_2']}\n"
            f"First sensor index: {mis_image_data['first_sensor_index']}"
        )

        ax.set_title(f"MIS image at {position} [counts]")

        ax.text(
            0.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=10, bbox=dict(facecolor="white", alpha=0.7)
        )

        self.canvas.draw()
