class PlotControl:
    def __init__(self, parent) -> None:
        self.parent = parent
        self.ui = parent.ui

    def set_live_plotting(self):
        self.parent.live_plot.set_plotting_mode(
            analysis_mode=self.ui.analysis_type_combobox.currentText(),
            positions_mode=self.ui.display_show_combobox.currentText(),
            units=self.ui.plot_units_combobox.currentText(),
            noise_show=self.ui.noise_show_combobox.currentText(),
        )
        self.parent.live_plot.resolution = max(
            self.ui.dut_counts_rev.value(),
            self.ui.ref_number_of_periods.value() * self.ui.ref_interpolation_factor.value(),
        )

    def set_constant_reading(self, check_state):
        self.parent.acquisition_worker.constant_reading = check_state

    def set_dut_direction(self, checkstate):
        self.parent.acquisition_worker.invert_dut = checkstate

    def on_zero_offset(self):
        self.parent.acquisition_worker.set_zero_offset = True

    def on_buffer_change(self, buffer):
        self.parent.live_plot.buffer_size = buffer

    def on_plot_save(self):
        self.parent.live_plot.save_plot()

    def record_data_continuously(self, check_state):
        if check_state:
            self.parent.acquisition_worker.recording_length = self.parent.max_acquired_samples
            self.parent.on_recording(True)

        else:
            self.parent.acquisition_worker.recording_stop = True

    def record_data_defined_samples(self):
        self.parent.acquisition_worker.recording_length = self.ui.record_samples_spinbox.value()
        self.parent.on_recording(True)
