import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pathlib import Path
from PyQt5.QtCore import pyqtSignal, QThread


class PlotAnalysis(QThread):
    finished_signal = pyqtSignal(str)

    def __init__(
        self, x, ref_scaled, dut_scaled, ref_counts, dut_counts, noise_analyse, saving_path: Path, plot_name: str
    ):
        super().__init__()
        self.x = x.copy()
        self.ref_scaled = ref_scaled.copy()
        self.dut_scaled = dut_scaled.copy()
        self.dut_counts = dut_counts.copy()
        self.ref_counts = ref_counts.copy()
        self.noise_analyse = noise_analyse
        self.saving_path = saving_path
        self.plot_name = plot_name or "plot"

    def run(self):
        self.analyse_data()
        self.finished_signal.emit("Saved successfully")

    def analyse_data(self):
        if self.noise_analyse:
            noise = self.dut_counts - np.mean(self.dut_counts)
            six_sig_noise_count = np.std(noise) * 6
            six_sig_noise_scaled = np.std(noise) * 6

            self.plot_data(
                x_data=None,
                y_data=[self.dut_counts - self.dut_counts[0]],
                label=["Error [deg]"],
                title=f"Position noise: ±{six_sig_noise_count:.3f} [counts], ±{six_sig_noise_scaled:.3f}° [deg]",
                x_label="Sample Index",
                y_label="Error [deg]",
                saving_path=self.saving_path / f"{self.plot_name}_analysis.png",
            )
        else:
            scaled_pos, scaled_ref = self.sort_positions(self.dut_scaled, self.ref_scaled)
            scaled_pos = scaled_pos - scaled_pos[0]

            error = scaled_pos - scaled_ref
            error = error - error[0]

            error_p2p = np.max(error) - np.min(error)
            rms = np.sqrt(np.mean(error**2))

            # POSITION FIGURE
            self.plot_data(
                x_data=scaled_ref,
                y_data=[scaled_ref, scaled_pos],
                label=["Reference [deg]", "DUT [deg]"],
                title="Encoder_position",
                x_label="Sample Index",
                y_label="Position [deg]",
                saving_path=self.saving_path / f"{self.plot_name}_position.png",
            )

            # ERROR FIGURE
            self.plot_data(
                x_data=scaled_ref,
                y_data=[error],
                label=["Error [deg]"],
                title=f"Encoder Error (P2P={error_p2p:.3f}°, RMS={rms:.3f}°)",
                x_label="Sample Index",
                y_label="Error [deg]",
                saving_path=self.saving_path / f"{self.plot_name}_analysis.png",
            )

    @staticmethod
    def sort_positions(scaled_position, scaled_reference):
        sorted_reference = np.sort(scaled_reference)
        sorted_position = scaled_position[np.argsort(scaled_reference)]
        sorted_position = np.unwrap(2 * np.pi * sorted_position / 360) / np.pi / 2 * 360
        return sorted_position, sorted_reference

    @staticmethod
    def plot_data(x_data, y_data, label, title, x_label, y_label, saving_path):
        fig = Figure(figsize=(12, 6), dpi=150)
        FigureCanvas(fig)
        ax = fig.add_subplot(111)

        for i, data in enumerate(y_data):
            if x_data is None:
                ax.plot(data, label=label[i])
            else:
                ax.plot(x_data, data, label=label[i])

        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
        ax.legend()

        fig.tight_layout()
        fig.savefig(saving_path)
        plt.close(fig)
