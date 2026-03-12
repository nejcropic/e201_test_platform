import traceback
from e201_test_platform import messages
from e201_test_platform.motor_worker import MotorWorker


class ManualMotor:
    def __init__(self, parent):
        self.parent = parent
        self.ui = parent.ui
        self.messages = messages
        self.motor_worker: MotorWorker = parent.motor_worker

    def initialize_motor(self, motor_config: dict):
        if not self.motor_worker.initialized:
            try:
                self.motor_worker.initialize_motor(motor_config)
                self.ui.debug_motor_widget.setDisabled(False)
                self.ui.motor_connect_button.setText("DISCONNECT")

            except Exception as e:
                tb = traceback.format_exc()
                self.messages.show_error(f"Error: {e}", f"Line: {tb}")
                self.ui.debug_motor_widget.setDisabled(True)
                self.ui.motor_connect_button.setText("CONNECT")

        else:
            try:
                self.call_motor_function("disconnect")
            except Exception as e:
                tb = traceback.format_exc()
                self.messages.show_error(f"Error: {e}", f"Line: {tb}")
                pass

            self.ui.debug_motor_widget.setDisabled(True)
            self.ui.motor_connect_button.setText("CONNECT")

    def call_motor_function(self, func_name: str, *args):
        self.motor_worker.enqueue_command(func_name, *args)
