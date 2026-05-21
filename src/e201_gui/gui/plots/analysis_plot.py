import numpy as np
import pyqtgraph as pg  # type: ignore


class AnalysisPlot:
    def __init__(self):
        # Define plot
        self.plot = pg.PlotItem(title="ANALYSIS")
        self.plot.setLabel("left", "Error", units="deg")
        self.plot.setLabel("bottom", "Sample Index")
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # Define curves
        self.err_curve = self.plot.plot(pen=pg.mkPen(color=(0, 200, 0), width=2))
        self.noise_curve = self.plot.plot(pen=pg.mkPen(color=(200, 0, 0), width=2))
        self.dnl_curve = self.plot.plot(pen=pg.mkPen(color=(0, 0, 200), width=2))
        self.inl_curve = self.plot.plot(
            pen=pg.mkPen(
                color=(0, 0, 0),
                width=2,
            )
        )

    def set_left_label(self, text: str, units: str):
        self.plot.setLabel("left", text, units=units)

    def update_plot(self, x, err, inl, dnl, noise_dut, noise_ref, analysis_mode, noise_show):
        # Set visible curves
        self.err_curve.setVisible(analysis_mode == "Error")
        self.inl_curve.setVisible(analysis_mode == "INL")
        self.dnl_curve.setVisible(analysis_mode == "DNL")
        self.noise_curve.setVisible(analysis_mode == "Noise")

        if noise_show == "DUT":
            noise = noise_dut
        else:
            noise = noise_ref

        # Set curve data
        self.err_curve.setData(x, err)
        self.inl_curve.setData(x, inl)
        self.dnl_curve.setData(x, dnl)
        self.noise_curve.setData(x, noise)

        # Set plot Y range
        if analysis_mode == "Noise":
            ymax = np.percentile(np.abs(noise), 95)
            ymax = max(ymax, 2.0)
        else:
            ymax = np.percentile(np.abs(err), 99)
            ymax = max(ymax, 0.0001)

        self.plot.setYRange(-ymax, ymax)
