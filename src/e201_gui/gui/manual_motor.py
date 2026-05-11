import traceback
from e201_gui.gui import messages


class ManualMotor:
    def __init__(self, parent):
        self.parent = parent
        self.ui = parent.ui
        self.messages = messages

    def initialize_motor(self):
        motor_type = self.ui.supported_motors.currentText()
        if not self.parent.motor_worker.initialized:
            try:
                self.parent.motor_worker.initialize_motor(motor_type)
                self.ui.debug_motor_widget.setDisabled(False)
                self.ui.motor_connect_button.setText("DISCONNECT")

            except Exception as e:
                tb = traceback.format_exc()
                self.messages.show_error(f"Error: {e}", f"Line: {tb}")
                self.ui.debug_motor_widget.setDisabled(True)
                self.ui.motor_connect_button.setText("CONNECT")

        else:
            try:
                self.call_motor_function("close_connection")
            except Exception as e:
                tb = traceback.format_exc()
                self.messages.show_error(f"Error: {e}", f"Line: {tb}")
                pass

            self.ui.debug_motor_widget.setDisabled(True)
            self.ui.motor_connect_button.setText("CONNECT")

    def call_motor_function(self, func_name: str, *args):
        self.parent.motor_worker.enqueue_command(func_name, *args)

    def on_enable_motor(self):
        if self.ui.enable_motor_checkbox.isChecked():
            self.call_motor_function("enable")
        else:
            self.call_motor_function("disable")
