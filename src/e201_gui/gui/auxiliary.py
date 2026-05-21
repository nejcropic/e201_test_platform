from pathlib import Path
from e201_gui.e201_drivers.parser import Parser
from e201_gui.e201_drivers.registers_preset import get_registers_preset
from e201_gui.gui import messages
from e201_gui.helpers.file_handlers import load_yaml, save_to_yaml
from letp_e2019.e2019 import get_all_e201
from e201_gui.motor_drivers.supported_motors import supported_motor_types


class Auxiliary:
    def __init__(self, parent):
        self.parent = parent
        self.ui = parent.ui
        self.messages = messages
        self.available_e201 = get_all_e201()
        self.preset_registers: dict = {}

    def populate_supported_motors(self):
        for motor_type in supported_motor_types.keys():
            self.ui.supported_motors.addItem(motor_type)

    def get_e201_devices(self):
        self.available_e201 = get_all_e201()

    def populate_comports(self):
        self.ui.dut_comport_combobox.clear()
        self.ui.ref_comport_combobox.clear()

        self.get_e201_devices()
        if any(self.available_e201):
            for device in self.available_e201:
                if device.e2019_type == self.ui.dut_type_groupbox.currentText():
                    self.ui.dut_comport_combobox.addItem(f"{device.e2019_comport} - {device.e2019_serial}")
                if device.e2019_type == self.ui.ref_type_groupbox.currentText():
                    self.ui.ref_comport_combobox.addItem(f"{device.e2019_comport} - {device.e2019_serial}")

        self.ui.dut_comport_combobox.addItem("None")
        self.ui.ref_comport_combobox.addItem("None")

        self.ui.dut_comport_combobox.setCurrentIndex(0)
        self.ui.ref_comport_combobox.setCurrentIndex(0)

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
            "is_rotary": True,
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
            self.save_last_settings()
        except Exception:
            self.messages.show_warning("Cannot initialize parser!")

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

    def load_register_access_preset(self):
        encoder = self.ui.predefined_registers.currentText()
        preset_values = get_registers_preset(encoder)
        self.preset_registers = preset_values
        self.update_load_register_combobox(preset_values)
        if self.parent.acquisition_worker.master.dut is not None:
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

    def load_last_settings(self):
        filepath = Path(self.parent.last_settings_filename)
        if not filepath.exists():
            return

        settings = load_yaml(self.parent.last_settings_filename)
        dut_set = settings.get("dut_settings")
        ref_set = settings.get("ref_settings")
        # DUT
        self.ui.dut_communication_combobox.setCurrentText(dut_set.get("communication").upper())
        self.ui.dut_counts_rev.setValue(dut_set.get("resolution"))
        self.ui.dut_singleturn_bits.setValue(dut_set.get("singleturn_bits"))
        self.ui.dut_multiturn_bits.setValue(dut_set.get("multiturn_bits"))
        self.ui.dut_status_bits.setValue(dut_set.get("status_bits"))
        self.ui.dut_crc_bits.setValue(dut_set.get("crc_bits"))
        self.ui.dut_bytes.setValue(dut_set.get("dut_bytes"))
        self.ui.dut_polarity.setValue(dut_set.get("polarity"))
        self.ui.dut_phase.setValue(dut_set.get("phase"))
        self.ui.dut_frequency.setValue(dut_set.get("frequency"))

        # REF
        self.ui.ref_interpolation_factor.setValue(ref_set.get("interpolation_factor"))
        self.ui.ref_number_of_periods.setValue(ref_set.get("number_of_periods"))

    def save_last_settings(self):
        encoder_data = {"dut_settings": self.get_dut_parameters(), "ref_settings": self.get_ref_parameters()}

        save_to_yaml(encoder_data, self.parent.last_settings_filename)

    def clear_error(self):
        self.ui.error_log.setText("--")
