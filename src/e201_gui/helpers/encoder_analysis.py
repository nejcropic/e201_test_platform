from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


class EncoderAnalysis:
    def __init__(self):
        pass

    def compute_error(self, dut_scaled, ref_scaled):
        error = dut_scaled - ref_scaled
        error -= error[0]

        return error

    def compute_metrics(self, error):
        p2p = np.max(error) - np.min(error)
        rms = np.sqrt(np.mean(error**2))
        return p2p, rms

    def compute_noise(self, dut_counts):
        noise = dut_counts - np.mean(dut_counts)
        return noise, np.std(noise) * 6

    def compute_inl(self, dut_scaled, ref_scaled):
        """
        INL = position error relative to ideal transfer.

        Returns:
            inl [deg]
        """

        inl = dut_scaled - ref_scaled
        inl -= np.mean(inl)

        return inl

    def compute_dnl(self, dut_scaled, ref_scaled):
        """
        DNL = step size error between adjacent samples.

        Returns:
            dnl [deg]
        """

        dut_step = np.diff(dut_scaled)
        ref_step = np.diff(ref_scaled)

        dnl = dut_step - ref_step
        dnl -= np.mean(dnl)

        return dnl

    @staticmethod
    def sort_positions(pos, ref):
        idx = np.argsort(ref)
        sorted_ref = ref[idx]
        sorted_pos = pos[idx]
        sorted_pos = np.unwrap(2 * np.pi * sorted_pos / 360) / (2 * np.pi) * 360
        return sorted_pos, sorted_ref


class Plotter:
    def __init__(self, saving_path: Path, plot_name: str) -> None:
        self.saving_path = saving_path
        self.plot_name = plot_name
        pass

    def plot_noise(self, noise, noise_sigma):
        self.plot_data(
            x_data=None,
            y_data=[noise],
            label=[f"Noise [counts]: ±{noise_sigma:.3f} [counts]"],
            title="Position noise",
            x_label="Sample Index",
            y_label="Noise [counts]",
            saving_path=self.saving_path / f"{self.plot_name}_analysis.png",
        )

    def plot_positions(self, scaled_ref, scaled_pos):
        self.plot_data(
            x_data=scaled_ref,
            y_data=[scaled_ref, scaled_pos],
            label=["Reference [deg]", "DUT [deg]"],
            title="Encoder_position",
            x_label="Sample Index",
            y_label="Position [deg]",
            saving_path=self.saving_path / f"{self.plot_name}_position.png",
        )

    def plot_error(self, scaled_ref, error, p2p, rms):
        self.plot_data(
            x_data=scaled_ref,
            y_data=[error],
            label=["Error [deg]"],
            title=f"Encoder Error (P2P={p2p:.3f}°, RMS={rms:.3f}°)",
            x_label="Sample Index",
            y_label="Error [deg]",
            saving_path=self.saving_path / f"{self.plot_name}_analysis.png",
        )

    def plot_inl(self, scaled_ref, inl):
        p2p = np.max(inl) - np.min(inl)
        rms = np.sqrt(np.mean(inl**2))

        self.plot_data(
            x_data=scaled_ref,
            y_data=[inl],
            label=[f"INL | P2P={p2p:.6f} deg | RMS={rms:.6f} deg"],
            title="Integral Nonlinearity (INL)",
            x_label="Reference Position [deg]",
            y_label="INL [deg]",
            saving_path=self.saving_path / f"{self.plot_name}_inl.png",
        )

    def plot_dnl(self, scaled_ref, dnl):
        p2p = np.max(dnl) - np.min(dnl)
        rms = np.sqrt(np.mean(dnl**2))

        self.plot_data(
            x_data=scaled_ref[:-1],
            y_data=[dnl],
            label=[f"DNL | P2P={p2p:.6f} deg | RMS={rms:.6f} deg"],
            title="Differential Nonlinearity (DNL)",
            x_label="Reference Position [deg]",
            y_label="DNL [deg]",
            saving_path=self.saving_path / f"{self.plot_name}_dnl.png",
        )

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
