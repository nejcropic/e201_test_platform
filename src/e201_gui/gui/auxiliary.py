import traceback
from e201_gui.e201_drivers.parser import Parser
from e201_gui.e201_drivers.registers_presets import get_registers_preset
from e201_gui.gui import messages
from letp_e2019.e2019 import get_all_e201


class Auxiliary:
    def __init__(self, parent):
        self.parent = parent
        self.ui = parent.ui
        self.messages = messages
        self.available_e201 = get_all_e201()
        self.preset_registers: dict = {}

    def get_e201_devices(self):
        self.available_e201 = get_all_e201()

    def populate_comports(self):
        self.ui.dut_comport_combobox.clear()
        self.ui.ref_comport_combobox.clear()

        self.get_e201_devices()
        if any(self.available_e201):
            for device in self.available_e201:
                if device.e2019_type == self.ui.dut_type_groupbox.currentText():
                    self.ui.dut_comport_combobox.addItem(f"{device.e2019_comport}")
                if device.e2019_type == self.ui.ref_type_groupbox.currentText():
                    self.ui.ref_comport_combobox.addItem(f"{device.e2019_comport}")

        self.ui.dut_comport_combobox.addItem("None")
        self.ui.ref_comport_combobox.addItem("None")

        self.ui.dut_comport_combobox.setCurrentIndex(0)
        self.ui.ref_comport_combobox.setCurrentIndex(0)

    def connect_dut(self):
        port = self.ui.dut_comport_combobox.currentText()
        e201_type = self.ui.dut_type_groupbox.currentText()
        voltage = 5000 if self.ui.five_volt_button.isChecked() else 3300
        if port != "None":
            if self.parent.acquisition_worker.master.dut is None:
                try:
                    self.update_parser()
                    self.call_reg_function("connect_dut", e201_type, port)
                    self.call_reg_function("dut_power_on", voltage)
                    self.call_reg_function("set_dut_communication", self.parent.acquisition_worker.parser.dut_settings)
                    self.set_connection_button(self.ui.dut_connection_indication, self.ui.dut_comport_connect, True)
                    self.load_register_access_preset()

                except Exception as e:
                    tb = traceback.format_exc()
                    self.messages.show_warning(
                        "Cannot connect to dut!",
                        f"Error: {e} \nLine: {tb}",
                    )
            else:
                self.sync_sampling(False)
                self.call_reg_function("close_dut")
                self.set_connection_button(self.ui.dut_connection_indication, self.ui.dut_comport_connect, False)

    def connect_ref(self):
        port = self.ui.ref_comport_combobox.currentText()
        e201_type = self.ui.ref_type_groupbox.currentText()
        if port != "None":
            if self.parent.acquisition_worker.master.ref is None:
                try:
                    self.update_parser()
                    self.call_reg_function("connect_ref", e201_type, port)
                    self.call_reg_function("ref_power_on", 5000)
                    self.set_connection_button(self.ui.ref_connection_indication, self.ui.ref_comport_connect, True)
                except Exception as e:
                    tb = traceback.format_exc()
                    self.messages.show_warning(
                        "Cannot connect to reference!",
                        f"Error: {e} \nLine: {tb}",
                    )
            else:
                self.sync_sampling(False)
                self.call_reg_function("close_ref")
                self.set_connection_button(self.ui.ref_connection_indication, self.ui.ref_comport_connect, False)

    def sync_sampling(self, checked):
        if checked:
            if self.parent.acquisition_worker.master.dut is None or self.parent.acquisition_worker.master.ref is None:
                self.messages.show_warning(
                    "Connection with DUT and reference have to be established to sync samples!",
                    "",
                )
                return

            self.call_reg_function("enable_synced_sampling")
            self.parent.on_zero_offset()

        else:
            self.call_reg_function("disable_synced_sampling")

    def get_dut_parameters(self):
        return {
            "communication": self.ui.dut_communication_combobox.currentText().lower(),
            "resolution": self.ui.dut_counts_rev.value(),
            "singleturn_bits": self.ui.dut_singleturn_bits.value(),
            "multiturn_bits": self.ui.dut_multiturn_bits.value(),
            "status_bits": self.ui.dut_status_bits.value(),
            "crc_bits": self.ui.dut_crc_bits.value(),
            "dut_bytes": self.ui.dut_bytes.value(),
            "polarity": self.ui.dut_polarity.value(),
            "phase": self.ui.dut_phase.value(),
            "frequency": self.ui.dut_frequency.value(),
            "is_rotary": self.ui.application_type_combobox.currentText() == "Rotary",
        }

    def get_ref_parameters(self):
        return {
            "interpolation_factor": self.ui.ref_interpolation_factor.value(),
            "number_of_periods": self.ui.ref_number_of_periods.value(),
        }

    def update_parser(self):
        encoder_data = {"dut_settings": self.get_dut_parameters(), "ref_settings": self.get_ref_parameters()}

        try:
            self.parent.acquisition_worker.parser = Parser(encoder_data=encoder_data)
        except Exception:
            self.messages.show_warning("Cannot initialize parser!")

    def write_registers(self):
        address, value, signed, bank, length = self.get_register_params()
        self.parent.acquisition_worker.enqueue_command("write_dut_register", value, bank, address, length, signed)

    def read_registers(self):
        address, value, signed, bank, length = self.get_register_params()
        self.parent.acquisition_worker.enqueue_command("read_dut_register", bank, address, length, signed)

    def call_reg_function(self, func_name: str, *args):
        self.parent.acquisition_worker.enqueue_command(func_name, *args)

    def set_multiturn(self):
        mt_value = self.ui.multiturn_value.value()
        self.parent.acquisition_worker.enqueue_command("set_multiturn", mt_value)

    def set_position_offset(self):
        offset_value = self.ui.position_offset_value.value()
        self.parent.acquisition_worker.enqueue_command("set_position_offset", offset_value)

    def get_register_params(self):
        address = self.ui.register_address_spinbox.value()
        value = self.ui.register_value_spinbox.value()
        signed = self.ui.register_signed_checkbox.isChecked()
        bank = self.ui.register_bank_spinbox.value()
        length = self.ui.register_length_spinbox.value()
        return address, value, signed, bank, length

    @staticmethod
    def set_connection_button(indicator, button, connected: bool = False):
        if connected:
            indicator.setText("● CONNECTED")
            indicator.setStyleSheet("color: #2E7D32; font-weight: 600;")
            button.setText("DISCONNECT")
        else:
            indicator.setText("● DISCONNECTED")
            indicator.setStyleSheet("color: #C62828; font-weight: 600;")
            button.setText("CONNECT")

    def dut_power_on(self):
        voltage = 5000 if self.ui.five_volt_button.isChecked() else 3300
        self.parent.acquisition_worker.enqueue_command("dut_power_on", voltage)

    def dut_power_cycle(self):
        voltage = 5000 if self.ui.five_volt_button.isChecked() else 3300
        self.parent.acquisition_worker.enqueue_command("dut_power_cycle", voltage)

    def load_register_access_preset(self):
        encoder = self.ui.predefined_registers.currentText()
        preset_values = get_registers_preset(encoder)
        self.preset_registers = preset_values
        self.update_load_register_combobox(preset_values)
        if self.parent.acquisition_worker.master.dut is None:
            return
        self.parent.acquisition_worker.enqueue_command("set_register_access", preset_values)

    def update_load_register_combobox(self, registers: dict):
        self.ui.loaded_registers.clear()
        for register in registers.keys():
            self.ui.loaded_registers.addItem(register)

    def update_current_register(self):
        selected_reg = self.ui.loaded_registers.currentText()
        register = self.preset_registers.get(selected_reg)
        if register is None:
            return

        self.ui.register_value_spinbox.setValue(register.get("value"))
        self.ui.register_address_spinbox.setValue(register.get("address"))
        self.ui.register_bank_spinbox.setValue(register.get("bank"))
        self.ui.register_length_spinbox.setValue(register.get("length"))
        self.ui.register_signed_checkbox.setCheckState(bool(register.get("is_signed")))
