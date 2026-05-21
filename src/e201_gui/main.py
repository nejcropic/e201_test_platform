import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt, pyqtSlot

from e201_gui.gui.acquisition_worker import AcquisitionWorker
from e201_gui.gui.connect_elements import ConnectElements
from e201_gui.e201_drivers.parser import Parser
from e201_gui.gui.device_handling import DeviceHandling
from e201_gui.gui.helpers import load_icon, load_pixmap
from e201_gui.gui.miss_image_window import MisImageWindow
from e201_gui.gui.plot_control import PlotControl
from e201_gui.gui.ui_template.e201_ui_template import Ui_MainWindow
from e201_gui.gui.auxiliary import Auxiliary
from e201_gui.gui.manual_motor import ManualMotor
from e201_gui.gui.motor_worker import MotorWorker
from e201_gui.gui.live_plot import LivePlot
from e201_gui.gui import messages


class E201TestPlatform(QtWidgets.QMainWindow):
    last_settings_filename = "last_settings.yaml"
    max_acquired_samples = 10000000

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("E201 Test Platform")

        self.plot_workers = []

        self.acquisition_worker: AcquisitionWorker = None  # type: ignore
        self.motor_worker: MotorWorker = None  # type: ignore
        self.messages = messages
        self.device_handling = DeviceHandling(self)
        self.plot_control = PlotControl(self)
        self.auxiliary = Auxiliary(self)
        encoder_data = {
            "dut_settings": self.auxiliary.get_dut_parameters(),
            "ref_settings": self.auxiliary.get_ref_parameters(),
        }
        self.parser = Parser(encoder_data)
        self.manual_motor = ManualMotor(self)
        self.live_plot = LivePlot(self)

    def _initialize(self):
        self.auxiliary.load_last_settings()
        self.auxiliary.populate_comports()
        self._init_threads()
        ConnectElements(self)
        self.plot_control.set_live_plotting()
        self.auxiliary.populate_supported_motors()
        self.auxiliary.load_register_access_preset()
        pixmap = load_pixmap("logo.png")
        self.ui.logo_label.setPixmap(pixmap)
        self.ui.logo_label.setPixmap(
            pixmap.scaled(
                self.ui.logo_label.size(),
                aspectRatioMode=Qt.KeepAspectRatio,  # type: ignore
                transformMode=Qt.SmoothTransformation,  # type: ignore
            )
        )

    def _init_threads(self):
        self.motor_worker_setup()
        self.acquisition_worker_setup()

        self.gui_timer = QtCore.QTimer(self)
        self.gui_timer.timeout.connect(self.live_plot.refresh_plots)
        self.gui_timer.start(33)

    def motor_worker_setup(self):
        self.motor_worker = MotorWorker()
        self.motor_worker.speed_signal.connect(self.on_speed_update)
        self.motor_worker.error_signal.connect(self.on_motor_stopped)
        self.motor_worker.start()

    def acquisition_worker_setup(self):
        self.acquisition_worker = AcquisitionWorker(self.live_plot.buffer, self.parser)
        self.acquisition_worker.error_signal.connect(self.handle_error)
        self.acquisition_worker.register_response_signal.connect(self.handle_register_response)
        self.acquisition_worker.device_info_signal.connect(self.handle_e201_info)
        self.acquisition_worker.miss_image_signal.connect(self.plot_miss_image)
        self.acquisition_worker.recording_finished_signal.connect(self.on_recording)
        self.acquisition_worker.start()

    @pyqtSlot(object)
    def handle_e201_info(self, e201_info):
        dut_info = e201_info.get("dut")
        ref_info = e201_info.get("ref")

        if dut_info is not None:
            self.ui.power_supply_dut.setText(f"{str(dut_info.get('power_supply'))}")
            self.ui.serial_number_dut.setText(f"{str(dut_info.get('serial_number'))}")
            self.ui.version_dut.setText(f"{str(dut_info.get('software_version'))}")
            self.ui.build_number_dut.setText(f"{str(dut_info.get('build_number'))}")

        if ref_info is not None:
            self.ui.power_supply_ref.setText(f"{str(ref_info.get('power_supply'))}")
            self.ui.serial_number_ref.setText(f"{str(ref_info.get('serial_number'))}")
            self.ui.version_ref.setText(f"{str(ref_info.get('software_version'))}")
            self.ui.build_number_ref.setText(f"{str(ref_info.get('build_number'))}")

    @pyqtSlot(object)
    def on_speed_update(self, speed):
        self.ui.current_speed_label.setText(f"{float(speed):.2f}")

    @pyqtSlot(object)
    def on_motor_stopped(self, error):
        self.motor_worker_setup()
        self.handle_error(error)

    @pyqtSlot(object)
    def handle_error(self, data):
        try:
            self.ui.error_log.setText(str(data))
            self.messages.show_error(str(data), "")
        except Exception as e:
            print(e)

    @pyqtSlot(object)
    def handle_register_response(self, data):
        self.ui.register_response_bin.setText(f"Raw: {data['response_raw']}")
        self.ui.register_response_int.setText(f"Int: {data['response_int']}")
        self.ui.register_response_hex.setText(f"Bin: {data['response_str']}")

    @pyqtSlot(object)
    def plot_miss_image(self, mis_image_data):
        last_position = self.acquisition_worker.last_position
        self.mis_window = MisImageWindow(mis_image_data, last_position["Position"])
        self.mis_window.show()

    @pyqtSlot(object)
    def on_recording(self, recording_state):
        if recording_state:
            self.ui.start_recording.setDisabled(True)
            self.ui.sampling_indicator.setText("● RECORDING...")
            self.ui.sampling_indicator.setStyleSheet("QLabel {color: #C62828; font-weight: 600;}")
            self.acquisition_worker.recorded_data = []
            self.acquisition_worker.recording = True

        else:
            self.ui.start_recording.setDisabled(False)
            self.ui.sampling_indicator.setText("● RECORDING FINISHED!")
            self.ui.sampling_indicator.setStyleSheet("QLabel {color: #2E7D32; font-weight: 600;}")

    def closeEvent(self, event):  # type: ignore
        try:
            if self.acquisition_worker is not None:
                self.acquisition_worker.stop_worker()
                self.acquisition_worker.wait(1500)
                self.acquisition_worker = None  # type: ignore
        except Exception:
            pass

        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)

    app.setFont(QtGui.QFont("Segoe UI", 9))
    app.setWindowIcon(load_icon("rls_logo.ico"))

    w = E201TestPlatform()
    w._initialize()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
