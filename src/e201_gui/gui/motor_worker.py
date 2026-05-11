import traceback
from queue import Queue, Empty
from PyQt5.QtCore import QThread, pyqtSignal
from e201_gui.motor_drivers.motor_base import MotorBase
from e201_gui.motor_drivers.supported_motors import get_supported_motor


class MotorWorker(QThread):
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(object)
    speed_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.loop_flag = True
        self.initialized = False
        self.motor: MotorBase = None  # type: ignore
        self.command_queue = Queue()

    def run(self):
        try:
            while self.loop_flag:
                self._loop_iteration()
                self.msleep(1)
        except Exception as e:
            self.error_signal.emit([e, traceback.format_exc()])
        finally:
            self.finished_signal.emit()

    def enqueue_command(self, command: str, *args):
        self.command_queue.put((command, args))

    def _loop_iteration(self):
        if not self.initialized:
            return
        self._handle_continuous_read()
        self._handle_commands()

    def _handle_continuous_read(self):
        speed = self.motor.get_velocity()
        self.speed_signal.emit(speed)

    def _handle_commands(self):
        try:
            command, args = self.command_queue.get_nowait()
            getattr(self, command)(*args)
        except Empty:
            pass

    def initialize_motor(self, motor_type: str):
        self.motor = get_supported_motor(motor_type)
        self.initialized = True
        self.enable()

    def enable(self):
        self.motor.enable()

    def disable(self):
        self.motor.disable()

    def set_speed(self, speed):
        self.motor.set_speed(speed)

    def read_speed(self):
        speed = self.motor.get_velocity()
        self.speed_signal.emit(speed)

    def stop(self):
        self.motor.stop()

    def close_connection(self):
        self.initialized = False
        self.motor.disable()
        self.motor.disconnect()
