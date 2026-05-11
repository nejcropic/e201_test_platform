from e201_gui.gui.auxiliary import Auxiliary
from e201_gui.gui.device_handling import DeviceHandling
from e201_gui.gui.plot_control import PlotControl


class ConnectElements:
    def __init__(self, parent):
        self.parent = parent
        self.ui = parent.ui
        self.auxiliary: Auxiliary = parent.auxiliary
        self.device_handling: DeviceHandling = parent.device_handling
        self.plot_control: PlotControl = parent.plot_control

        self.connect_dut_elements()
        self.register_access_elements()
        self.connect_plot_control_elements()
        self.connect_motor_elements()
        self.power_elements()

    def connect_dut_elements(self):
        self.ui.update_parser_button.clicked.connect(self.auxiliary.update_parser)
        self.ui.dut_comport_connect.clicked.connect(self.device_handling.connect_dut)
        self.ui.ref_comport_connect.clicked.connect(self.device_handling.connect_ref)
        self.ui.refresh_ports_button.clicked.connect(self.auxiliary.populate_comports)
        self.ui.dut_type_groupbox.currentIndexChanged.connect(self.auxiliary.populate_comports)
        self.ui.ref_type_groupbox.currentIndexChanged.connect(self.auxiliary.populate_comports)
        self.ui.set_dut_communication.clicked.connect(
            lambda: self.device_handling.call_reg_function("set_dut_communication", self.auxiliary.get_dut_parameters())
        )

    def register_access_elements(self):
        self.ui.constant_reading.clicked.connect(self.plot_control.set_constant_reading)
        self.ui.predefined_registers.activated.connect(self.auxiliary.load_register_access_preset)
        self.ui.write_register_button.clicked.connect(self.device_handling.write_registers)
        self.ui.read_register_button.clicked.connect(self.device_handling.read_registers)
        self.ui.loaded_registers.activated.connect(self.auxiliary.update_current_register)
        self.ui.set_multiturn.clicked.connect(self.device_handling.set_multiturn)
        self.ui.set_position_offset.clicked.connect(self.device_handling.set_position_offset)
        self.ui.read_miss_image.clicked.connect(lambda: self.device_handling.call_reg_function("read_miss_image"))
        self.ui.start_calibration.clicked.connect(self.device_handling.start_self_calibration)
        self.ui.save_to_flash.clicked.connect(lambda: self.device_handling.call_reg_function("save_to_flash"))
        self.ui.factory_reset.clicked.connect(lambda: self.device_handling.call_reg_function("factory_reset"))

    def connect_plot_control_elements(self):
        self.ui.display_show_combobox.currentIndexChanged.connect(self.plot_control.set_live_plotting)
        self.ui.invert_dut_direction_checkbox.clicked.connect(self.plot_control.set_dut_direction)
        self.ui.plot_units_combobox.currentIndexChanged.connect(self.plot_control.set_live_plotting)
        self.ui.analysis_type_combobox.currentIndexChanged.connect(self.plot_control.set_live_plotting)
        self.ui.noise_show_combobox.currentIndexChanged.connect(self.plot_control.set_live_plotting)
        self.ui.zero_offset_button.clicked.connect(self.plot_control.on_zero_offset)
        self.ui.plot_buffer_size.valueChanged.connect(self.plot_control.on_buffer_change)
        self.ui.save_plot_button.clicked.connect(self.plot_control.on_plot_save)
        self.ui.record_data_checkbox.clicked.connect(self.plot_control.record_data_continuously)
        self.ui.start_recording.clicked.connect(self.plot_control.record_data_defined_samples)

    def power_elements(self):
        self.ui.dut_power_on.clicked.connect(self.device_handling.dut_power_on)
        self.ui.dut_power_cycle.clicked.connect(self.device_handling.dut_power_cycle)
        self.ui.dut_power_off.clicked.connect(lambda: self.device_handling.call_reg_function("dut_power_off"))
        self.ui.ref_power_on.clicked.connect(lambda: self.device_handling.call_reg_function("ref_power_on", 5000))
        self.ui.ref_power_off.clicked.connect(lambda: self.device_handling.call_reg_function("ref_power_off"))
        self.ui.ref_power_cycle.clicked.connect(lambda: self.device_handling.call_reg_function("ref_power_cycle", 5000))

    def connect_motor_elements(self):
        self.ui.motor_connect_button.clicked.connect(self.parent.manual_motor.initialize_motor)
        self.ui.set_speed_button.clicked.connect(
            lambda: self.parent.manual_motor.call_motor_function("set_speed", self.ui.speed_set.value())
        )
        self.ui.stop_button.clicked.connect(lambda: self.parent.manual_motor.call_motor_function("stop"))
        self.ui.enable_motor_checkbox.clicked.connect(self.parent.manual_motor.on_enable_motor)
