import sys
import pyqtgraph as pg
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot

from e201_test_platform.acquisition_worker import AcquisitionWorker
from e201_test_platform.connect_elements import ConnectElements
from e201_test_platform.gui.ui_template import Ui_MainWindow
from e201_test_platform.gui_auxiliary import GuiAuxiliary
from e201_test_platform.manual_motor import ManualMotor
from e201_test_platform.motor_worker import MotorWorker


class E201TestPlatform(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle('E201 Test Platform')

        self.set_zero_offset = False
        self.acquisition_worker: AcquisitionWorker = None # type: ignore
        self.gui_auxiliary = GuiAuxiliary(self)

        self.gui_auxiliary.populate_comports()

        self.motor_worker = MotorWorker()
        self.manual_motor = ManualMotor(self)
        self.motor_worker_setup()
        self._init_plots()
        self._init_threads()
        ConnectElements(self)

    def motor_worker_setup(self):
        self.motor_worker.finished_signal.connect(self.on_motor_stopped)
        self.motor_worker.speed_signal.connect(self.on_speed_update)
        self.motor_worker.start()

    def _init_threads(self):
        self.acquisition_worker = AcquisitionWorker()
        self.acquisition_worker.sample_signal.connect(self.on_sample)
        self.acquisition_worker.detailed_status_signal.connect(self.handle_detailed_status)
        self.acquisition_worker.error_signal.connect(self.on_worker_error)
        # self.acquisition_worker.finished_signal.connect(self.on_worker_finished)
        self.acquisition_worker.register_response_signal.connect(self.handle_register_response)

        self.acquisition_worker.start()

        self.gui_timer = QtCore.QTimer()
        self.gui_timer.timeout.connect(self.refresh_plots)
        self.gui_timer.start(33)  # ~30 Hz

    def _init_plots(self):
        self.buffer_size = 1000
        self.sample_counter = 0
        self.write_index = 0
        self.full = False
        self.error_offset = 0.0
        self.offset_samples_needed = 200  # averaging window
        self.offset_buffer = []
        self.offset_locked = False

        # Preallocated ring buffers
        self.x_data = np.zeros(self.buffer_size, dtype=np.float64)
        self.ref_data_angle = np.zeros(self.buffer_size, dtype=np.float64)
        self.dut_data_angle = np.zeros(self.buffer_size, dtype=np.float64)
        self.err_data_angle = np.zeros(self.buffer_size, dtype=np.float64)
        self.ref_data_counts = np.zeros(self.buffer_size, dtype=np.float64)
        self.dut_data_counts = np.zeros(self.buffer_size, dtype=np.float64)
        self.err_data_counts = np.zeros(self.buffer_size, dtype=np.float64)

        # ===== Position Plot =====
        self.position_plot = pg.PlotWidget(title="POSITIONS")
        self.position_plot.addLegend()
        self.position_plot.setLabel("left", "Position", units="deg")
        self.position_plot.setLabel("bottom", "Sample Index")

        self.ref_curve_angle = self.position_plot.plot(
            pen=pg.mkPen(color=(0, 200, 0), width=2),
            name="Reference [deg]"
        )

        self.dut_curve_angle = self.position_plot.plot(
            pen=pg.mkPen(color=(200, 0, 0), width=2),
            name="DUT [deg]"
        )

        self.ref_curve_counts = self.position_plot.plot(
            pen=pg.mkPen(color=(0, 0, 200), width=2),
            name="Reference [counts]"
        )

        self.dut_curve_counts = self.position_plot.plot(
            pen=pg.mkPen(color=(255, 255, 0), width=2),
            name="DUT [counts]"
        )

        # ===== Error Plot =====
        self.analysis_plot = pg.PlotWidget(title="ANALYSIS PLOT")
        self.analysis_plot.setLabel("left", "Error", units="deg")
        self.analysis_plot.setLabel("bottom", "Sample Index")

        self.err_curve_angle = self.analysis_plot.plot(
            pen=pg.mkPen(color=(0, 120, 255), width=2)
        )

        self.err_curve_counts = self.analysis_plot.plot(
            pen=pg.mkPen(color=(255, 120, 0), width=2)
        )

        # Performance settings
        for p in (self.position_plot, self.analysis_plot):
            p.showGrid(x=True, y=True)
            p.setDownsampling(mode='peak')
            p.setClipToView(True)

        self.ref_curve_angle.setVisible(True)
        self.dut_curve_angle.setVisible(True)
        self.err_curve_angle.setVisible(True)

        self.ui.position_plot_layout.addWidget(self.position_plot)
        self.ui.analysis_plot_layout.addWidget(self.analysis_plot)

    def on_sample(self, data: tuple):
        t, dut, ref = data
        if dut is not None:
            dut_angle = dut.get('angle', 0.0)
            dut_counts = dut.get('singleturn', 0)

            if self.ui.invert_dut_direction_checkbox.isChecked():
                res = self.acquisition_worker.dut.resolution
                dut_counts = (-dut_counts) % res
                dut_angle = dut_counts * 360.0 / res
                dut['angle'] = dut_angle

            self.ui.dut_position_hex_label.setText(f"DUT position [hex]: {str(dut.get('raw_hex'))}")
            self.ui.dut_counts_label.setText(f"DUT counts: {str(dut.get('singleturn'))}")
            self.ui.dut_position_label.setText(f"DUT position [deg]: {round(dut_angle, 3)}")

        if ref is not None:
            self.ui.ref_position_hex_label.setText(f"REF position [hex]: {str(ref.get('raw_hex'))}")
            self.ui.ref_counts_label.setText(f"REF counts: {str(ref.get('singleturn'))}")
            self.ui.ref_position_label.setText(f"REF position [deg]: {str(round(ref.get('angle'),3))}")

        self.update_plot((t, dut, ref))

    def update_plot(self, data: tuple):
        t, dut, ref = data

        dut_angle, dut_counts, status = 0.0, 0.0, None
        ref_angle, ref_counts = 0.0, 0.0

        if dut is not None:
            dut_angle = dut.get('angle', 0.0)
            dut_counts = dut.get('singleturn', 0.0)
            status = dut.get('status', None)

        if ref is not None:
            ref_angle = ref.get('angle', 0.0)
            ref_counts = ref.get('singleturn', 0.0)

        self.set_status_label(status)

        raw_error_deg = self._wrap_error_deg(dut_angle - ref_angle)

        if self.set_zero_offset:
            self.error_offset = raw_error_deg
            self.set_zero_offset = False

        error = self._wrap_error_deg(raw_error_deg - self.error_offset)

        if self.ui.noise_show_combobox.currentText() == "DUT":
            noise = self._wrap_error_deg(dut_counts - self.dut_data_counts[0])
        else:
            noise = self._wrap_error_deg(ref_counts - self.ref_data_counts[0])

        self.x_data[self.write_index] = self.sample_counter
        self.ref_data_angle[self.write_index] = ref_angle
        self.dut_data_angle[self.write_index] = dut_angle
        self.err_data_angle[self.write_index] = error
        self.ref_data_counts[self.write_index] = ref_counts
        self.dut_data_counts[self.write_index] = dut_counts
        self.err_data_counts[self.write_index] = noise

        self.write_index += 1

        if self.write_index >= self.buffer_size:
            self.write_index = 0
            self.full = True

        self.sample_counter += 1

    def set_status_label(self, status):
        if status is None:
            self.ui.dut_status_label.setStyleSheet("QLabel {\n"
                                                   "background-color: grey;\n"
                                                   "color: white;\n"
                                                   "border-radius: 10px;\n}")
            return

        self.ui.dut_status_label.setText(f"STATUS: {status}")
        if status < 2:
            self.ui.dut_status_label.setStyleSheet("QLabel {\nbackground-color: red;\n"
                                                   "border-radius: 10px;\ncolor: white;\n}")
        elif status == 2:
            self.ui.dut_status_label.setStyleSheet("QLabel {\nbackground-color: yellow;\n"
                                                   "border-radius: 10px;\ncolor: black;\n}")
        elif status == 3:
            self.ui.dut_status_label.setStyleSheet("QLabel {\nbackground-color: green;\n"
                                                   "border-radius: 10px;\ncolor: white;\n}")


    def refresh_plots(self):
        if self.write_index == 0 and not self.full:
            return

        if not self.full:
            x = self.x_data[:self.write_index]
            ref_deg = self.ref_data_angle[:self.write_index]
            dut_deg = self.dut_data_angle[:self.write_index]
            err_deg = self.err_data_angle[:self.write_index]
            ref_counts = self.ref_data_counts[:self.write_index]
            dut_counts = self.dut_data_counts[:self.write_index]
            err_counts = self.err_data_counts[:self.write_index]
        else:
            idx = np.arange(self.write_index, self.write_index + self.buffer_size) % self.buffer_size
            x = self.x_data[idx]
            ref_deg = self.ref_data_angle[idx]
            dut_deg = self.dut_data_angle[idx]
            err_deg = self.err_data_angle[idx]
            ref_counts = self.ref_data_counts[idx]
            dut_counts = self.dut_data_counts[idx]
            err_counts = self.err_data_counts[idx]

        mode, unit = self.get_display_mode()

        if unit == "Degrees":
            self.ref_curve_angle.setVisible(mode in ("DUT&REF", "REF"))
            self.dut_curve_angle.setVisible(mode in ("DUT&REF", "DUT"))
            self.ref_curve_counts.setVisible(False)
            self.dut_curve_counts.setVisible(False)
        elif unit == "Counts":
            self.ref_curve_counts.setVisible(mode in ("DUT&REF", "REF"))
            self.dut_curve_counts.setVisible(mode in ("DUT&REF", "DUT"))
            self.ref_curve_angle.setVisible(False)
            self.dut_curve_angle.setVisible(False)

        if self.ui.analysis_type_combobox.currentText() == "Noise":
            self.err_curve_counts.setVisible(True)
            self.err_curve_angle.setVisible(False)
            max_err = 5
        else:
            self.err_curve_counts.setVisible(False)
            self.err_curve_angle.setVisible(True)
            max_err = np.max(np.abs(err_deg))
            if max_err < 1:
                max_err = 1
            # max_err = np.max(np.abs(err_deg)) * 1.2

        self.analysis_plot.setYRange(-max_err, max_err)

        p2p = np.max(err_deg) - np.min(err_deg)
        rms = np.sqrt(np.mean(err_deg ** 2))
        self.ui.p2p_error_label.setText(f"P2P: {p2p:.3f} [deg]")
        self.ui.rms_error_label.setText(f"RMS: {rms:.3f} [deg]")

        self.ref_curve_angle.setData(x, ref_deg)
        self.ref_curve_counts.setData(x, ref_counts)
        self.dut_curve_angle.setData(x, dut_deg)
        self.dut_curve_counts.setData(x, dut_counts)
        self.err_curve_angle.setData(x, err_deg)
        self.err_curve_counts.setData(x, err_counts)

    @staticmethod
    def _wrap_error_deg(err: float) -> float:
        return (err + 180.0) % 360.0 - 180.0

    def handle_detailed_status(self, data: str):
        self.ui.dut_detailed_status_label.setText(f"DUT detailed status: {data}")

    def on_buffer_change(self, buffer):
        self.buffer_size = buffer

    def handle_register_response(self, data: dict):
        pass
        # print("Register response:", data)
        # self.ui.response_textedit.append(str(data))

    def on_stop(self):
        if self.acquisition_worker is None:
            return
        self.acquisition_worker.stop_worker()
        self.append_log("Stopping worker...")

    def on_worker_error(self, response):
        print(response)

    def get_display_mode(self) -> (str, str):
        unit = self.ui.plot_units_combobox.currentText().strip()
        analysis  = self.ui.analysis_type_combobox.currentText()
        err_unit = "Counts" if analysis == "Noise" else "Degrees"
        if analysis == "Noise":
            self.ui.noise_show_combobox.setEnabled(True)
        else:
            self.ui.noise_show_combobox.setEnabled(False)

        self.analysis_plot.setLabel("left", analysis, units=f"{err_unit}")
        self.position_plot.setLabel("left", "Position", units=f"{unit}")
        return self.ui.display_show_combobox.currentText().strip(), unit

    def on_disconnect(self):
        if self.acquisition_worker is None:
            return
        try:
            self.acquisition_worker.disconnect()
        except Exception:
            pass
        # self.acquisition_worker.stop_worker()
        self.append_log("Disconnect requested (closing ports)...")

    def initialize_motor(self):
        motor_config = {"type": "Rotacijska_epos_naprava"}
        self.manual_motor.initialize_motor(motor_config)

    def on_zero_offset(self):
        self.set_zero_offset = True

    def on_speed_update(self, speed: float):
        self.ui.current_speed_label.setText(f"{speed:.2f}")

    def on_enable_motor(self):
        if self.ui.enable_motor_checkbox.isChecked():
            self.manual_motor.call_motor_function("enable")
        else:
            self.manual_motor.call_motor_function("disable")

    @pyqtSlot(object)
    def on_motor_stopped(self, error):
        self.motor_worker = MotorWorker()
        self.motor_worker_setup()

    def closeEvent(self, event):
        try:
            if self.acquisition_worker is not None:
                self.acquisition_worker.stop_worker()
                self.acquisition_worker.wait(1500)
                self.acquisition_worker = None
        except Exception:
            pass

        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)

    app.setFont(QtGui.QFont("Segoe UI", 9))

    w = E201TestPlatform()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()