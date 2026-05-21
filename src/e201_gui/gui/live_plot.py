import numpy as np
from pathlib import Path

import pyqtgraph as pg  # type: ignore
from PyQt5 import QtGui

from e201_gui.helpers.plot_analysis import PlotAnalysis
from e201_gui.gui.position_buffer import PositionBuffer
from e201_gui.helpers.encoder_analysis import EncoderAnalysis

from e201_gui.gui.plots.position_plot import PositionPlot
from e201_gui.gui.plots.analysis_plot import AnalysisPlot
from e201_gui.gui.plots.index_plot_window import IndexPlotWindow


class LivePlot:
    def __init__(self, parent):

        self.parent = parent
        self.ui = parent.ui

        self.buffer = PositionBuffer()

        self.buffer_size = self.ui.plot_buffer_size.value()

        self.error_offset = 0.0
        self._last_error = 0.0

        self.offset_samples_needed = 200
        self.offset_buffer = []
        self.offset_locked = False

        self.recorded_data = []
        self.recording = False

        self.analysis_mode = "Error"
        self.positions_mode = "DUT"
        self.plot_units = "Degrees"
        self.noise_show = "DUT"

        self.last_sample = None
        self._last_status = None

        self.resolution = 360

        self.encoder_analysis = EncoderAnalysis()

        # Popup windows
        self.index_window = None

        # Main graphics widget
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground("w")

        # Position plot
        self.position_plot = PositionPlot()
        self.plot_widget.addItem(self.position_plot.plot)

        # Analysis plot
        self.plot_widget.nextRow()

        self.analysis_plot = AnalysisPlot()

        self.plot_widget.addItem(self.analysis_plot.plot)

        # Shared plot settings
        font = QtGui.QFont()
        font.setPointSize(10)

        for p in (
            self.position_plot.plot,
            self.analysis_plot.plot,
        ):
            p.setDownsampling(mode="peak")
            p.setClipToView(True)

            p.getAxis("left").setStyle(tickFont=font)

            p.getAxis("bottom").setStyle(tickFont=font)

            p.getAxis("left").setPen("k")
            p.getAxis("bottom").setPen("k")

        self.ui.plot_layout.addWidget(self.plot_widget)

        self.saving_path = Path("saved_plots")
        self.saving_path.mkdir(exist_ok=True)

    def set_plotting_mode(self, analysis_mode: str, positions_mode: str, units: str, noise_show: str):

        self.analysis_mode = analysis_mode
        self.positions_mode = positions_mode
        self.plot_units = units
        self.noise_show = noise_show

        err_unit = "Counts" if analysis_mode == "Noise" else "Degrees"

        self.ui.noise_show_combobox.setEnabled(analysis_mode == "Noise")

        self.analysis_plot.set_left_label(analysis_mode, err_unit)

        self.position_plot.set_left_label("Position", units)

    def open_index_plot(self):

        if self.index_window is None:
            self.index_window = IndexPlotWindow()

        self.index_window.show()

    def refresh_plots(self):

        snap = self.buffer.snapshot(self.buffer_size)

        if snap is None:
            return

        x = snap["sample_idx"]

        dut_counts = snap["dut_counts"]
        ref_counts = snap["ref_counts"]

        dut_deg = snap["dut_deg"]
        ref_deg = snap["ref_deg"]

        err = snap["err_deg"]
        inl = snap["inl_deg"]
        dnl = snap["dnl_deg"]

        noise_dut = snap["noise_dut"]
        noise_ref = snap["noise_ref"]

        dut_index = snap["dut_index"]

        self.position_plot.update_plot(
            x=x,
            ref_deg=ref_deg,
            dut_deg=dut_deg,
            ref_counts=ref_counts,
            dut_counts=dut_counts,
            units=self.plot_units,
            positions_mode=self.positions_mode,
            resolution=self.resolution,
        )

        self.analysis_plot.update_plot(
            x=x,
            err=err,
            inl=inl,
            dnl=dnl,
            noise_dut=noise_dut,
            noise_ref=noise_ref,
            analysis_mode=self.analysis_mode,
            noise_show=self.noise_show,
        )

        if self.index_window is not None:
            self.index_window.update_plot(x, dut_index)

        p2p, rms = self.encoder_analysis.compute_metrics(err)

        _, stdev = self.encoder_analysis.compute_noise(dut_counts)

        try:
            latest = self.parent.acquisition_worker.latest_sample

            self.update_ui(
                {
                    "dut_counts": int(dut_counts[-1]),
                    "ref_counts": int(ref_counts[-1]),
                    "dut_scaled": float(dut_deg[-1]),
                    "ref_scaled": float(ref_deg[-1]),
                    "dut_index": latest.get("dut_index"),
                    "ref_index": latest.get("ref_index"),
                    "status": latest.get("status"),
                    "p2p": float(p2p),
                    "rms": float(rms),
                    "inl": float(inl[-1]),
                    "stdev": float(stdev),
                    "multiturn": latest.get("multiturn"),
                }
            )

        except Exception:
            pass

    def update_ui(self, d):

        round_pos = 5

        self.ui.dut_counts_label.setText(f"DUT [counts]: {d['dut_counts']}")
        self.ui.dut_position_label.setText(f"DUT [deg]: {d['dut_scaled']:.{round_pos}f}")
        self.ui.ref_counts_label.setText(f"REF [counts]: {d['ref_counts']}")
        self.ui.ref_position_label.setText(f"REF [deg]: {d['ref_scaled']:.{round_pos}f}")
        self.ui.dut_index_label.setText(f"DUT Index: {d['dut_index']}")
        self.ui.ref_index_label.setText(f"Error DNL: {d['inl']:.{round_pos}f} [deg]")
        self.ui.p2p_error_label.setText(f"Error P2P: {d.get('p2p'):.{round_pos}f} [deg]")
        self.ui.rms_error_label.setText(f"Error RMS: {d.get('rms'):.{round_pos}f} [deg]")
        if self.analysis_mode != "Noise":
            self.ui.stdev_noise_label.setText("Noise P2P (6σ): -- [counts]")
        else:
            self.ui.stdev_noise_label.setText(f"Noise P2P (6σ): {d.get('stdev'):.{round_pos}f} [counts]")

        mt = d.get("multiturn")
        if mt is None:
            mt = "--"

        self.ui.multiturn_label.setText(f"DUT Multiturn: {mt}")

        status = d.get("status", None)

        if status != self._last_status:
            if status is None:
                self.ui.dut_status_label.setStyleSheet(
                    "QLabel {\nbackground-color: grey;\ncolor: white;\nborder-radius: 10px;\n}"
                )

                return

            self.ui.dut_status_label.setText(f"STATUS: {status}")

            if status < 2:
                self.ui.dut_status_label.setStyleSheet(
                    "QLabel {\nbackground-color: red;\nborder-radius: 10px;\ncolor: white;\n}"
                )

            elif status == 2:
                self.ui.dut_status_label.setStyleSheet(
                    "QLabel {\nbackground-color: yellow;\nborder-radius: 10px;\ncolor: black;\n}"
                )

            elif status == 3:
                self.ui.dut_status_label.setStyleSheet(
                    "QLabel {\nbackground-color: green;\nborder-radius: 10px;\ncolor: white;\n}"
                )

        self._last_status = status

    @staticmethod
    def _wrap_error_deg(err: float) -> float:
        return (err + 180.0) % 360.0 - 180.0

    def save_plot(self):

        plot_name = self.ui.save_plot_name.text().strip()

        data = self.parent.acquisition_worker.recorded_data

        if not data:
            print("No data recorded!")
            return

        arr = np.array(
            [
                [
                    d["x"],
                    d["ts"],
                    d["ref_counts"],
                    d["dut_counts"],
                    d["ref_deg"],
                    d["dut_deg"],
                    d["ref_index"],
                    d["dut_index"],
                ]
                for d in data
            ]
        )

        x = arr[:, 0]
        ts = arr[:, 1]

        ref_counts = arr[:, 2]
        dut_counts = arr[:, 3]

        ref_scaled = arr[:, 4]
        dut_scaled = arr[:, 5]

        ref_index = arr[:, 6]
        dut_index = arr[:, 7]

        self.save_to_csv(
            arr,
            plot_name,
        )

        noise_analyse = self.ui.analysis_type_combobox.currentText() == "Noise"

        worker = PlotAnalysis(
            x=x,
            ts=ts,
            ref_scaled=ref_scaled,
            dut_scaled=dut_scaled,
            ref_index=ref_index,
            dut_index=dut_index,
            ref_counts=ref_counts,
            dut_counts=dut_counts,
            noise_analyse=noise_analyse,
            saving_path=self.saving_path,
            plot_name=plot_name,
        )

        self.parent.plot_workers.append(worker)

        worker.finished_signal.connect(
            lambda msg, w=worker: self.on_plot_finished(
                w,
                msg,
            )
        )

        worker.start()

    def on_plot_finished(self, worker, msg):

        worker.quit()
        worker.wait()

        self.parent.plot_workers.remove(worker)

    def save_to_csv(self, arr, plot_name):

        csv_path = self.saving_path / f"{plot_name}_raw.csv"

        np.savetxt(
            csv_path,
            arr,
            delimiter=",",
            header=("sample_idx,timestamp,ref_counts,dut_counts,ref_deg,dut_deg,ref_index,dut_index"),
            comments="",
        )
