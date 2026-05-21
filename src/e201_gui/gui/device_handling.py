import traceback


class DeviceHandling:
    def __init__(self, parent) -> None:
        self.parent = parent
        self.ui = parent.ui

    def connect_dut(self):
        port = self.ui.dut_comport_combobox.currentText()
        e201_type = self.ui.dut_type_groupbox.currentText()
        voltage = 5000 if self.ui.five_volt_button.isChecked() else 3300
        if self.parent.acquisition_worker.master.dut is None:
            if port != "None":
                try:
                    port = port.split(" - ")[0]
                    self.parent.auxiliary.update_parser()
                    self.call_reg_function("connect_dut", e201_type, port)
                    self.call_reg_function("dut_power_on", voltage)
                    self.call_reg_function("set_dut_communication", self.parent.acquisition_worker.parser.dut_settings)
                    self.call_reg_function("get_e201_info")
                    self.parent.auxiliary.set_connection_button(
                        self.ui.dut_connection_indication, self.ui.dut_comport_connect, True
                    )
                    self.parent.auxiliary.load_register_access_preset()

                except Exception as e:
                    tb = traceback.format_exc()
                    self.parent.messages.show_warning("Cannot connect to dut!", f"Error: {e} \nLine: {tb}")
        else:
            self.call_reg_function("close_dut")
            self.parent.auxiliary.set_connection_button(
                self.ui.dut_connection_indication, self.ui.dut_comport_connect, False
            )

    def connect_ref(self):
        port = self.ui.ref_comport_combobox.currentText()
        e201_type = self.ui.ref_type_groupbox.currentText()
        if self.parent.acquisition_worker.master.ref is None:
            if port != "None":
                try:
                    port = port.split(" - ")[0]
                    self.parent.auxiliary.update_parser()
                    self.call_reg_function("connect_ref", e201_type, port)
                    self.call_reg_function("ref_power_on", 5000)
                    self.call_reg_function("get_e201_info")
                    self.parent.auxiliary.set_connection_button(
                        self.ui.ref_connection_indication, self.ui.ref_comport_connect, True
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    self.parent.messages.show_warning(
                        "Cannot connect to reference!",
                        f"Error: {e} \nLine: {tb}",
                    )
        else:
            self.call_reg_function("close_ref")
            self.parent.auxiliary.set_connection_button(
                self.ui.ref_connection_indication, self.ui.ref_comport_connect, False
            )

    def get_e201_info(self):
        self.parent.acquisition_worker.enqueue_command("get_e201_info")

    def write_registers(self):
        address, value, signed, bank, length = self.parent.auxiliary.get_register_params()
        self.parent.acquisition_worker.enqueue_command("write_dut_register", value, bank, address, length, signed)

    def read_registers(self):
        address, value, signed, bank, length = self.parent.auxiliary.get_register_params()
        self.parent.acquisition_worker.enqueue_command("read_dut_register", bank, address, length, signed)

    def call_reg_function(self, func_name: str, *args):
        self.parent.acquisition_worker.enqueue_command(func_name, *args)

    def set_multiturn(self):
        mt_value = self.ui.multiturn_value.value()
        self.parent.acquisition_worker.enqueue_command("set_multiturn", mt_value)

    def set_position_offset(self):
        offset_value = self.ui.position_offset_value.value()
        self.parent.acquisition_worker.enqueue_command("set_position_offset", offset_value)

    def dut_power_on(self):
        voltage = 5000 if self.ui.five_volt_button.isChecked() else 3300
        self.parent.acquisition_worker.enqueue_command("dut_power_on", voltage)

    def dut_power_cycle(self):
        voltage = 5000 if self.ui.five_volt_button.isChecked() else 3300
        self.parent.acquisition_worker.enqueue_command("dut_power_cycle", voltage)

    def start_self_calibration(self):
        current_speed = self.ui.speed_set.value()
        if current_speed == 0:
            raise ValueError("Self calibration not possible when motor is stable!")

        selfcal_timeout = 60 / current_speed * 1.3 + 0.5  # calculate one rotation time and increase by 30%
        self.parent.acquisition_worker.enqueue_command("start_self_calibration", selfcal_timeout)
