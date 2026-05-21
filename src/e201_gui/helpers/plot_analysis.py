from e201_gui.helpers.encoder_analysis import EncoderAnalysis, Plotter
from pathlib import Path
from PyQt5.QtCore import pyqtSignal, QThread


class PlotAnalysis(QThread):
    finished_signal = pyqtSignal(str)

    def __init__(
        self,
        x,
        ts,
        ref_scaled,
        dut_scaled,
        ref_counts,
        dut_counts,
        dut_index,
        ref_index,
        noise_analyse,
        saving_path: Path,
        plot_name: str,
    ):
        super().__init__()
        self.x = x.copy()
        self.ts = ts.copy()
        self.dut_scaled = dut_scaled.copy()
        self.ref_scaled = ref_scaled.copy()
        self.dut_counts = dut_counts.copy()
        self.ref_counts = ref_counts.copy()
        self.dut_index = dut_index.copy()
        self.ref_index = ref_index.copy()
        self.noise_analyse = noise_analyse
        self.saving_path = saving_path
        self.plot_name = plot_name or "plot"

    def run(self):
        self.analyse_data()
        self.finished_signal.emit("Saved successfully")

    def analyse_data(self):
        analysis = EncoderAnalysis()
        plotter = Plotter(self.saving_path, self.plot_name)
        if self.noise_analyse:
            noise, noise_sigma = analysis.compute_noise(self.dut_counts)
            plotter.plot_noise(noise=noise, noise_sigma=noise_sigma)

        else:
            scaled_pos, scaled_ref = analysis.sort_positions(self.dut_scaled, self.ref_scaled)
            error = analysis.compute_error(scaled_pos, scaled_ref)
            p2p, rms = analysis.compute_metrics(error)
            dnl = analysis.compute_dnl(scaled_pos, scaled_ref)
            inl = analysis.compute_inl(scaled_pos, scaled_ref)

            # POSITION FIGURE
            plotter.plot_positions(scaled_ref=scaled_ref, scaled_pos=scaled_pos)

            # DNL FIGURE
            plotter.plot_dnl(scaled_ref=scaled_ref, dnl=dnl)

            # INL FIGURE
            plotter.plot_inl(scaled_ref=scaled_ref, inl=inl)

            # ERROR FIGURE
            plotter.plot_error(scaled_ref=scaled_ref, error=error, p2p=p2p, rms=rms)

            # INDEX FIGURE
            scaled_dut_index, scaled_ref = analysis.sort_positions(self.dut_index, self.ref_scaled)
            scaled_ref_index, scaled_ref = analysis.sort_positions(self.dut_index, self.ref_scaled)
            plotter.plot_index(scaled_ref=scaled_ref, scaled_index=scaled_dut_index, index_label="DUT")
            plotter.plot_index(scaled_ref=scaled_ref, scaled_index=scaled_ref_index, index_label="REF")
