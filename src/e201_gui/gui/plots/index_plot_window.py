import pyqtgraph as pg  # type: ignore


class IndexPlotWindow:
    def __init__(self):
        # Define widget
        self.widget = pg.GraphicsLayoutWidget()
        self.widget.setWindowTitle("INDEX PLOT")
        self.widget.resize(900, 500)
        self.widget.setBackground("w")

        # Define plot
        self.plot = self.widget.addPlot(title="Z INDEX")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Index error")
        self.plot.setLabel("bottom", "Sample index")

        # Define curves
        self.curve = self.plot.plot(pen=pg.mkPen(color=(200, 0, 200), width=2))

    def show(self):
        self.widget.show()

    def update_plot(self, x, y):
        self.curve.setData(x, y)
