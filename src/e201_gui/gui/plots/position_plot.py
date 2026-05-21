import pyqtgraph as pg  # type: ignore


class PositionPlot:
    def __init__(self):
        # Define plot
        self.plot = pg.PlotItem(title="POSITIONS")
        self.plot.addLegend()
        self.plot.setLabel("left", "Position", units="deg")
        self.plot.setLabel("bottom", "Sample Index")
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # Define curves
        self.ref_curve_angle = self.plot.plot(pen=pg.mkPen(color=(0, 200, 0), width=2), name="Reference [deg]")
        self.dut_curve_angle = self.plot.plot(pen=pg.mkPen(color=(200, 0, 0), width=2), name="DUT [deg]")
        self.ref_curve_counts = self.plot.plot(pen=pg.mkPen(color=(0, 200, 0), width=2), name="Reference [counts]")
        self.dut_curve_counts = self.plot.plot(pen=pg.mkPen(color=(200, 0, 0), width=2), name="DUT [counts]")

    def set_left_label(self, text: str, units: str):
        self.plot.setLabel("left", text, units=units)

    def update_plot(self, x, ref_deg, dut_deg, ref_counts, dut_counts, units, positions_mode, resolution):
        show_deg = units == "Degrees"
        show_counts = units == "Counts"

        # Set visible curves
        self.ref_curve_angle.setVisible(show_deg and positions_mode in ("DUT&REF", "REF"))
        self.dut_curve_angle.setVisible(show_deg and positions_mode in ("DUT&REF", "DUT"))
        self.ref_curve_counts.setVisible(show_counts and positions_mode in ("DUT&REF", "REF"))
        self.dut_curve_counts.setVisible(show_counts and positions_mode in ("DUT&REF", "DUT"))

        # Set curve data
        self.ref_curve_angle.setData(x, ref_deg)
        self.dut_curve_angle.setData(x, dut_deg)
        self.ref_curve_counts.setData(x, ref_counts)
        self.dut_curve_counts.setData(x, dut_counts)

        # Set plot Y range
        if units == "Degrees":
            self.plot.setYRange(0, 360)

        else:
            self.plot.setYRange(0, resolution)
