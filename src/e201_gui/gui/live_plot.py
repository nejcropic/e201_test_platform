import numpy as np
from pathlib import Path
import pyqtgraph as pg
from PyQt5 import QtGui
from e201_gui.helpers.plot_analysis import PlotAnalysis
from e201_gui.gui.position_buffer import PositionBuffer


class LivePlot:
    def __init__(self, parent):
        self.parent = parent
        self.ui = parent.ui
        self.buffer = PositionBuffer()
        self.buffer_size = 1000
        self.error_offset = 0.0
        self._last_error = 0.0
        self.offset_samples_needed = 200  # averaging window
        self.offset_buffer = []
        self.offset_locked = False
        self.recorded_data = []
        self.recording = False
        self.analysis_mode = "Error"
        self.positions_mode = "DUT"
        self.plot_units = "DEG"
        self.last_sample = None
        self._last_status = None

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground("w")
        # ===== Position Plot =====
        self.position_plot = self.plot_widget.addPlot(title="POSITIONS")
        self.position_plot.addLegend()
        self.position_plot.setLabel("left", "Position", units="deg")
        self.position_plot.setLabel("bottom", "Sample Index")

        # Curves
        self.ref_curve_angle = self.position_plot.plot(pen=pg.mkPen(color=(0, 200, 0), width=2), name="Reference [deg]")
        self.dut_curve_angle = self.position_plot.plot(pen=pg.mkPen(color=(200, 0, 0), width=2), name="DUT [deg]")
        self.ref_curve_counts = self.position_plot.plot(
            pen=pg.mkPen(color=(0, 0, 200), width=2), name="Reference [counts]"
        )
        self.dut_curve_counts = self.position_plot.plot(pen=pg.mkPen(color=(255, 255, 0), width=2), name="DUT [counts]")

        # ===== Error Plot =====
        self.plot_widget.nextRow()
        self.analysis_plot = self.plot_widget.addPlot(title="ANALYSIS PLOT")
        self.analysis_plot.setLabel("left", "Error", units="deg")
        self.analysis_plot.setLabel("bottom", "Sample Index")

        self.err_curve = self.analysis_plot.plot(pen=pg.mkPen(color=(0, 200, 0), width=2))
        self.noise_curve = self.analysis_plot.plot(pen=pg.mkPen(color=(200, 0, 0), width=2))
        self.dnl_curve = self.analysis_plot.plot(pen=pg.mkPen(color=(0, 0, 200), width=2))
        self.inl_curve = self.analysis_plot.plot(pen=pg.mkPen(color=(0, 0, 0), width=2))

        self.position_plot.getAxis("left").setPen("k")
        self.position_plot.getAxis("bottom").setPen("k")

        # Performance settings
        font = QtGui.QFont()
        font.setPointSize(10)
        for p in (self.position_plot, self.analysis_plot):
            p.showGrid(x=True, y=True, alpha=0.3)
            p.setDownsampling(mode="peak")
            p.setClipToView(True)
            p.getAxis("left").setStyle(tickFont=font)
            p.getAxis("bottom").setStyle(tickFont=font)

        self.ui.plot_layout.addWidget(self.plot_widget)

        self.saving_path = Path("saved_plots")
        self.saving_path.mkdir(exist_ok=True)

    def set_plotting_mode(self, analysis_mode: str, positions_mode: str, units: str):
        self.analysis_mode = analysis_mode
        self.positions_mode = positions_mode
        self.plot_units = units

        err_unit = "Counts" if analysis_mode == "Noise" else "Degrees"
        if analysis_mode == "Noise":
            self.ui.noise_show_combobox.setEnabled(True)
        else:
            self.ui.noise_show_combobox.setEnabled(False)

        self.analysis_plot.setLabel("left", analysis_mode, units=f"{err_unit}")
        self.position_plot.setLabel("left", "Position", units=f"{units}")

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
        noise = snap["noise"]

        # plot visibility
        show_deg = self.plot_units == "Degrees"
        show_counts = self.plot_units == "Counts"

        self.ref_curve_angle.setVisible(show_deg and self.positions_mode in ("DUT&REF", "REF"))
        self.dut_curve_angle.setVisible(show_deg and self.positions_mode in ("DUT&REF", "DUT"))
        self.ref_curve_counts.setVisible(show_counts and self.positions_mode in ("DUT&REF", "REF"))
        self.dut_curve_counts.setVisible(show_counts and self.positions_mode in ("DUT&REF", "DUT"))

        self.err_curve.setVisible(self.analysis_mode == "Error")
        self.inl_curve.setVisible(self.analysis_mode == "INL")
        self.dnl_curve.setVisible(self.analysis_mode == "DNL")
        self.noise_curve.setVisible(self.analysis_mode == "Noise")

        # set data
        self.ref_curve_angle.setData(x, ref_deg)
        self.dut_curve_angle.setData(x, dut_deg)
        self.ref_curve_counts.setData(x, ref_counts)
        self.dut_curve_counts.setData(x, dut_counts)

        self.err_curve.setData(x, err)
        self.inl_curve.setData(x, inl)
        self.dnl_curve.setData(x, dnl)
        self.noise_curve.setData(x, noise)

        # axis scaling
        if self.plot_units == "Degrees":
            self.position_plot.setYRange(0, 360)

        if self.analysis_mode == "Noise":
            ymax = np.percentile(np.abs(noise), 95)
            ymax = max(ymax, 2.0)
            self.analysis_plot.setYRange(-ymax, ymax)

        elif self.analysis_mode == "DNL":
            ymax = np.percentile(np.abs(dnl), 95)
            ymax = max(ymax, 0.0001)
            self.analysis_plot.setYRange(-ymax, ymax)

        else:
            ymax = np.percentile(np.abs(err), 99)
            ymax = max(ymax, 0.0001)
            self.analysis_plot.setYRange(-ymax, ymax)

        # stats
        p2p = np.max(err) - np.min(err)
        rms = np.sqrt(np.mean(err**2))

        try:
            latest = self.parent.acquisition_worker.latest_sample
            self.update_ui(
                {
                    "dut_counts": int(dut_counts[-1]),
                    "ref_counts": int(ref_counts[-1]),
                    "dut_scaled": float(dut_deg[-1]),
                    "ref_scaled": float(ref_deg[-1]),
                    "status": latest.get("status"),
                    "p2p": float(p2p),
                    "rms": float(rms),
                    "multiturn": latest.get("multiturn"),
                }
            )
        except Exception:
            pass

    def update_ui(self, d):
        round_pos = 5
        self.ui.dut_counts_label.setText(f"DUT counts: {d['dut_counts']}")
        self.ui.dut_position_label.setText(f"DUT position [deg]: {d['dut_scaled']:.{round_pos}f}")

        self.ui.ref_counts_label.setText(f"REF counts: {d['ref_counts']}")
        self.ui.ref_position_label.setText(f"REF position [deg]: {d['ref_scaled']:.{round_pos}f}")

        self.ui.p2p_error_label.setText(f"P2P: {d.get('p2p'):.{round_pos}f} [deg]")

        mt = d.get("multiturn")
        if mt is None:
            mt = "--"
        self.ui.multiturn_label.setText(f"Multiturn: {mt}")

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

        if data:
            arr = np.array(
                [
                    [
                        d["x"],
                        d["ref_counts"],
                        d["dut_counts"],
                        d["ref_deg"],
                        d["dut_deg"],
                    ]
                    for d in data
                ]
            )

            x = arr[:, 0]
            ref_counts = arr[:, 1]
            dut_counts = arr[:, 2]
            ref_scaled = arr[:, 3]
            dut_scaled = arr[:, 4]

            noise_analyse = True if self.ui.analysis_type_combobox.currentText() == "Noise" else False
            worker = PlotAnalysis(
                x=x,
                ref_scaled=ref_scaled,
                dut_scaled=dut_scaled,
                ref_counts=ref_counts,
                dut_counts=dut_counts,
                noise_analyse=noise_analyse,
                saving_path=self.saving_path,
                plot_name=plot_name,
            )

            self.parent.plot_workers.append(worker)

            worker.finished_signal.connect(lambda msg, w=worker: self.parent.on_plot_finished(w, msg))

            worker.start()

        else:
            print("No data recorded!")
