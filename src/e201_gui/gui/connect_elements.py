from e201_gui.gui.auxiliary import Auxiliary


class ConnectElements:
    def __init__(self, parent):
        self.parent = parent
        self.ui = parent.ui
        self.auxiliary: Auxiliary = parent.auxiliary

        self.connect_dut_elements()
        self.register_access_elements()
        self.connect_plot_control_elements()
        self.connect_motor_elements()
        self.power_elements()

    def connect_dut_elements(self):
        self.ui.update_parser_button.clicked.connect(self.auxiliary.update_parser)
        self.ui.dut_comport_connect.clicked.connect(self.auxiliary.connect_dut)
        self.ui.ref_comport_connect.clicked.connect(self.auxiliary.connect_ref)
        self.ui.refresh_ports_button.clicked.connect(self.auxiliary.populate_comports)
        self.ui.dut_type_groupbox.currentIndexChanged.connect(self.auxiliary.populate_comports)
        self.ui.ref_type_groupbox.currentIndexChanged.connect(self.auxiliary.populate_comports)
        self.ui.set_dut_communication.clicked.connect(
            lambda: self.auxiliary.call_reg_function("set_dut_communication", self.auxiliary.get_dut_parameters())
        )

    def register_access_elements(self):
        self.ui.predefined_registers.activated.connect(self.auxiliary.load_register_access_preset)
        self.ui.write_register_button.clicked.connect(self.auxiliary.write_registers)
        self.ui.read_register_button.clicked.connect(self.auxiliary.read_registers)
        self.ui.loaded_registers.activated.connect(self.auxiliary.update_current_register)
        self.ui.set_multiturn.clicked.connect(self.auxiliary.set_multiturn)
        self.ui.set_position_offset.clicked.connect(self.auxiliary.set_position_offset)

    def connect_plot_control_elements(self):
        self.ui.display_show_combobox.currentIndexChanged.connect(self.parent.set_live_plotting)
        self.ui.plot_units_combobox.currentIndexChanged.connect(self.parent.set_live_plotting)
        self.ui.analysis_type_combobox.currentIndexChanged.connect(self.parent.set_live_plotting)
        self.ui.zero_offset_button.clicked.connect(self.parent.on_zero_offset)
        self.ui.plot_buffer_size.valueChanged.connect(self.parent.on_buffer_change)
        self.ui.save_plot_button.clicked.connect(self.parent.on_plot_save)
        self.ui.record_data_checkbox.clicked.connect(self.parent.record_data)

    def power_elements(self):
        self.ui.dut_power_on.clicked.connect(self.auxiliary.dut_power_on)
        self.ui.dut_power_cycle.clicked.connect(self.auxiliary.dut_power_cycle)
        self.ui.dut_power_off.clicked.connect(lambda: self.auxiliary.call_reg_function("dut_power_off"))
        self.ui.ref_power_on.clicked.connect(lambda: self.auxiliary.call_reg_function("ref_power_on", 5000))
        self.ui.ref_power_off.clicked.connect(lambda: self.auxiliary.call_reg_function("ref_power_off"))
        self.ui.ref_power_cycle.clicked.connect(lambda: self.auxiliary.call_reg_function("ref_power_cycle", 5000))

    def connect_motor_elements(self):
        self.ui.motor_connect_button.clicked.connect(self.parent.initialize_motor)
        self.ui.set_speed_button.clicked.connect(
            lambda: self.parent.manual_motor.call_motor_function("set_speed", self.ui.speed_set.value())
        )
        self.ui.stop_button.clicked.connect(lambda: self.parent.manual_motor.call_motor_function("soft_stop"))
        self.ui.enable_motor_checkbox.clicked.connect(self.parent.on_enable_motor)
