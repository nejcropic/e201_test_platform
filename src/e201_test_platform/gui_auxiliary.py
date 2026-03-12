import traceback
from e201_test_platform import messages
from e201_test_platform.e201_versions.e201b import E201B
from e201_test_platform.e201_versions.e201q import E201Q
from e201_test_platform.helpers import find_stm_electronics


class GuiAuxiliary:
    def __init__(self, parent):
        self.parent = parent
        self.ui = parent.ui
        self.messages = messages

    def populate_comports(self):
        e201_ports = find_stm_electronics()
        if any(e201_ports):
            self.ui.dut_comport_combobox.clear()
            self.ui.ref_comport_combobox.clear()
            self.ui.dut_comport_combobox.setEnabled(True)
            self.ui.ref_comport_combobox.setEnabled(True)
            for port in e201_ports:
                self.ui.dut_comport_combobox.addItem(port)
                self.ui.ref_comport_combobox.addItem(port)

    def connect_dut(self):
        port = self.ui.dut_comport_combobox.currentText()
        e201_type = self.ui.dut_type_groupbox.currentText()
        if port != "None":
            if self.parent.acquisition_worker.dut is None:
                parameters = self.get_dut_parameters()
                try:
                    self.call_reg_function('initialize_dut', e201_type, port, parameters)
                    self.set_connection_button(self.ui.dut_connection_indication, self.ui.dut_comport_connect, True)
                except Exception as e:
                    tb = traceback.format_exc()
                    self.messages.show_warning(
                        "Cannot connect to dut!",
                        f"Error: {e} \nLine: {tb}",
                    )
            else:
                self.ui.sync_reading_button.setCheckState(False)
                self.sync_sampling(False)
                self.call_reg_function('close_dut')
                self.set_connection_button(self.ui.dut_connection_indication, self.ui.dut_comport_connect, False)

    def connect_ref(self):
        port = self.ui.ref_comport_combobox.currentText()
        e201_type = self.ui.ref_type_groupbox.currentText()
        if port != "None":
            if self.parent.acquisition_worker.ref is None:
                parameters = self.get_ref_parameters()
                try:
                    self.call_reg_function('initialize_ref', e201_type, port, parameters)
                    self.set_connection_button(self.ui.ref_connection_indication, self.ui.ref_comport_connect, True)
                except Exception as e:
                    tb = traceback.format_exc()
                    self.messages.show_warning(
                        "Cannot connect to reference!",
                        f"Error: {e} \nLine: {tb}",
                    )
            else:
                self.sync_sampling(False)
                self.call_reg_function('close_ref')
                self.set_connection_button(self.ui.ref_connection_indication, self.ui.ref_comport_connect, False)

    def sync_sampling(self, checked):
        if checked:
            if self.parent.acquisition_worker.dut is None or self.parent.acquisition_worker.ref is None:
                self.messages.show_warning(
                    "Connection with DUT and reference have to be established to sync samples!", f"",)
                return

            self.call_reg_function('enable_synced_sampling')
            self.parent.on_zero_offset()

        else:
            if self.parent.acquisition_worker.dut is None or self.parent.acquisition_worker.ref is None:
                self.messages.show_warning(
                    "Connection with DUT and reference have to be established to sync samples!", f"",)
                return

            self.call_reg_function('disable_synced_sampling')

    def get_dut_parameters(self):
        return {
            'resolution': self.ui.dut_counts_rev.value(),
            'singleturn_bits': self.ui.dut_singleturn_bits.value(),
            'multiturn_bits': self.ui.dut_multiturn_bits.value(),
            'status_bits': self.ui.dut_status_bits.value(),
            'crc_bits': self.ui.dut_crc_bits.value(),
        }

    def get_ref_parameters(self):
        return {
            'resolution': self.ui.ref_counts_rev.value(),
            'singleturn_bits': self.ui.ref_singleturn_bits.value()
        }

    def write_registers(self):
        address, value, signed, bank, length = self.get_register_params()
        self.parent.acquisition_worker.enqueue_command('write_dut_register', value, bank, address, length, signed)

    def read_registers(self):
        address, value, signed, bank, length = self.get_register_params()
        self.parent.acquisition_worker.enqueue_command('read_dut_register', value, bank, address, length, signed)

    def write_registers_param(self,  value: int, param: str):
        self.parent.acquisition_worker.enqueue_command('write_dut_register_param', value, param)

    def read_registers_param(self,  value: int, param: str):
        self.parent.acquisition_worker.enqueue_command('read_dut_register_param', value, param)

    def call_reg_function(self, func_name: str, *args):
        self.parent.acquisition_worker.enqueue_command(func_name, *args)

    def get_register_params(self):
        address = self.ui.register_address.value()
        value = self.ui.register_value.value()
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
