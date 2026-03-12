from e201_test_platform.gui_auxiliary import GuiAuxiliary


class ConnectElements:
    def __init__(self, parent):
        self.parent = parent
        self.ui = parent.ui
        self.gui_auxiliary: GuiAuxiliary = parent.gui_auxiliary

        self.connect_dut_elements()
        self.register_access_elements()
        self.connect_plot_control_elements()
        self.connect_motor_elements()

    def connect_dut_elements(self):
        self.ui.dut_comport_connect.clicked.connect(self.gui_auxiliary.connect_dut)
        self.ui.ref_comport_connect.clicked.connect(self.gui_auxiliary.connect_ref)
        self.ui.refresh_ports_button.clicked.connect(self.gui_auxiliary.populate_comports)

    def register_access_elements(self):
        self.ui.write_register_button.clicked.connect(self.gui_auxiliary.write_registers)
        self.ui.read_register_button.clicked.connect(self.gui_auxiliary.read_registers)
        self.ui.set_multiturn_button.clicked.connect(lambda:
            self.gui_auxiliary.call_reg_function('set_multiturn', self.ui.set_multiturn_value.value()))
        self.ui.set_position_offset_buton.clicked.connect(lambda:
            self.gui_auxiliary.call_reg_function('set_position_offset', self.ui.set_position_offset_value.value()))
        self.ui.read_detailed_status_buton.clicked.connect(lambda:
            self.gui_auxiliary.call_reg_function('read_detailed_status'))


    def connect_plot_control_elements(self):
        self.ui.sync_reading_button.clicked.connect(self.parent.gui_auxiliary.sync_sampling)
        self.ui.display_show_combobox.currentIndexChanged.connect(self.parent.get_display_mode)
        self.ui.plot_units_combobox.currentIndexChanged.connect(self.parent.get_display_mode)
        self.ui.zero_offset_button.clicked.connect(self.parent.on_zero_offset)
        self.ui.plot_buffer_size.valueChanged.connect(self.parent.on_buffer_change)

    def connect_motor_elements(self):
        self.ui.motor_connect_button.clicked.connect(self.parent.initialize_motor)
        self.ui.set_speed_button.clicked.connect(
            lambda: self.parent.manual_motor.call_motor_function(
                "set_speed", self.ui.speed_set.value()
            )
        )
        self.ui.stop_button.clicked.connect(
            lambda: self.parent.manual_motor.call_motor_function("soft_stop")
        )
        self.ui.enable_motor_checkbox.clicked.connect(self.parent.on_enable_motor)