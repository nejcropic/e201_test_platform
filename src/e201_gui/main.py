import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot

from e201_gui.gui.acquisition_worker import AcquisitionWorker
from e201_gui.gui.connect_elements import ConnectElements
from e201_gui.e201_drivers.parser import Parser
from e201_gui.gui.ui_template.e201_ui_template import Ui_MainWindow
from e201_gui.gui.auxiliary import Auxiliary
from e201_gui.gui.manual_motor import ManualMotor
from e201_gui.gui.motor_worker import MotorWorker

from e201_gui.gui.live_plot import LivePlot


class E201TestPlatform(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("E201 Test Platform")

        self.set_zero_offset = False
        self.plot_workers = []
        self._last_status = None
        self.acquisition_worker: AcquisitionWorker = None  # type: ignore
        self.auxiliary = Auxiliary(self)
        encoder_data = {
            "dut_settings": self.auxiliary.get_dut_parameters(),
            "ref_settings": self.auxiliary.get_ref_parameters(),
        }
        self.parser = Parser(encoder_data)

        self.auxiliary.populate_comports()

        self.motor_worker = MotorWorker()
        self.manual_motor = ManualMotor(self)
        self.motor_worker_setup()
        self.live_plot = LivePlot(self)
        self._init_threads()
        ConnectElements(self)
        self.set_live_plotting()

    def motor_worker_setup(self):
        self.motor_worker.finished_signal.connect(self.on_motor_stopped)
        self.motor_worker.speed_signal.connect(self.on_speed_update)
        self.motor_worker.start()

    def _init_threads(self):
        self.acquisition_worker = AcquisitionWorker(self.live_plot.buffer, self.parser)
        self.acquisition_worker.error_signal.connect(self.on_worker_error)
        self.acquisition_worker.register_response_signal.connect(self.handle_register_response)
        self.acquisition_worker.start()

        self.gui_timer = QtCore.QTimer(self)
        self.gui_timer.timeout.connect(self.live_plot.refresh_plots)
        self.gui_timer.start(33)

    def record_data(self, check_state):
        if check_state:
            self.acquisition_worker.recording = True
            self.acquisition_worker.recorded_data = []
        else:
            self.acquisition_worker.recording = False

    def set_live_plotting(self):
        self.live_plot.set_plotting_mode(
            analysis_mode=self.ui.analysis_type_combobox.currentText(),
            positions_mode=self.ui.display_show_combobox.currentText(),
            units=self.ui.plot_units_combobox.currentText(),
        )

    def update_ui(self, d):
        self.ui.dut_position_hex_label.setText(f"DUT position [hex]: {d['dut_hex']}")
        self.ui.dut_counts_label.setText(f"DUT counts: {d['dut_counts']}")
        self.ui.dut_position_label.setText(f"DUT position [deg]: {d['dut_scaled']:.3f}")

        self.ui.ref_position_hex_label.setText(f"REF position [hex]: {d['ref_hex']}")
        self.ui.ref_counts_label.setText(f"REF counts: {d['ref_counts']}")
        self.ui.ref_position_label.setText(f"REF position [deg]: {d['ref_scaled']:.3f}")

        self.ui.p2p_error_label.setText(f"P2P: {d.get('p2p'):.3f} [deg]")
        self.ui.rms_error_label.setText(f"RMS: {d.get('rms'):.3f} [deg]")

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

    def on_buffer_change(self, buffer):
        self.live_plot.buffer_size = buffer

    def handle_register_response(self, data: dict):
        try:
            self.ui.register_response_bin.setText(f"{data['response_raw']}")
            self.ui.register_response_int.setText(f"{data['response_int']}")
            self.ui.register_response_hex.setText(f"{data['response_str']}")
        except Exception as e:
            print(e)
            pass

    def on_stop(self):
        if self.acquisition_worker is None:
            return
        self.acquisition_worker.stop_worker()
        self.append_log("Stopping worker...")

    def on_worker_error(self, response):
        print(response)

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
        self.acquisition_worker.set_zero_offset = True

    def on_speed_update(self, speed: float):
        self.ui.current_speed_label.setText(f"{speed:.2f}")

    def on_enable_motor(self):
        if self.ui.enable_motor_checkbox.isChecked():
            self.manual_motor.call_motor_function("enable")
        else:
            self.manual_motor.call_motor_function("disable")

    def on_plot_save(self):
        self.live_plot.save_plot()

    @pyqtSlot(object)
    def on_motor_stopped(self, error):
        self.motor_worker = MotorWorker()
        self.motor_worker_setup()

    def on_plot_finished(self, worker, msg):
        print(msg)
        worker.quit()
        worker.wait()
        self.plot_workers.remove(worker)

    def handle_detailed_status(self, data: str):
        self.ui.dut_detailed_status_label.setText(f"DUT detailed status: {data}")

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
