import sys
import time
from dataclasses import dataclass
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


# =========================
# Mock devices
# =========================

class MockMotor:
    """Mock motor producing speed [deg/s] and angle [deg]."""
    def __init__(self):
        self.speed_deg_s = 30.0
        self.enabled = True
        self._angle_deg = 0.0
        self._last_t = time.perf_counter()

    def set_speed(self, speed_deg_s: float):
        self.speed_deg_s = float(speed_deg_s)

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)

    def step(self):
        """Advance internal angle based on speed and dt."""
        now = time.perf_counter()
        dt = now - self._last_t
        self._last_t = now

        if self.enabled:
            self._angle_deg = (self._angle_deg + self.speed_deg_s * dt) % 360.0

        return self._angle_deg, self.speed_deg_s if self.enabled else 0.0


class MockEncoder:
    """
    Mock encoder returning counts.
    - reference has low noise
    - dut has higher noise + small periodic error component
    """
    def __init__(self, counts_per_rev: int, kind: str):
        self.cpr = int(counts_per_rev)
        self.kind = kind
        self._rng = np.random.default_rng(123 if kind == "ref" else 999)

    def read_counts_from_angle(self, angle_deg: float) -> int:
        # Ideal counts
        ideal = (angle_deg / 360.0) * self.cpr

        if self.kind == "ref":
            noise = self._rng.normal(0.0, 0.8)  # ~1 count noise
            error = 0.0
        else:
            noise = self._rng.normal(0.0, 3.0)  # more noise
            # periodic error (e.g., eccentricity / harmonic)
            error = 8.0 * np.sin(np.deg2rad(angle_deg * 2.0)) + 2.0 * np.sin(np.deg2rad(angle_deg * 7.0))

        counts = int(np.round(ideal + error + noise))
        return counts


# =========================
# Helpers (signal processing)
# =========================

@dataclass
class DecoderConfig:
    singleturn_bits: int = 12
    multiturn_bits: int = 0
    status_bits: int = 0
    crc_bits: int = 0
    counts_per_rev: int = 4096

    def wrap_counts(self, counts: int) -> int:
        cpr = max(1, int(self.counts_per_rev))
        return int(counts) % cpr

    def counts_to_deg(self, counts: int) -> float:
        cpr = max(1, int(self.counts_per_rev))
        return (counts % cpr) / cpr * 360.0


class RingBuffer:
    """Simple ring buffer for numeric arrays."""
    def __init__(self, maxlen: int):
        self.maxlen = int(maxlen)
        self._x = np.zeros(self.maxlen, dtype=float)
        self._n = 0

    def append(self, value: float):
        idx = self._n % self.maxlen
        self._x[idx] = value
        self._n += 1

    def array(self) -> np.ndarray:
        n = min(self._n, self.maxlen)
        if n == 0:
            return np.array([], dtype=float)

        start = (self._n - n) % self.maxlen
        if start + n <= self.maxlen:
            return self._x[start:start + n].copy()
        else:
            part1 = self._x[start:].copy()
            part2 = self._x[: (start + n) % self.maxlen].copy()
            return np.concatenate([part1, part2], axis=0)

    def clear(self):
        self._n = 0


def wrap_error_deg(err_deg: np.ndarray) -> np.ndarray:
    """
    Wrap error to [-180, +180) domain.
    Useful when working with wrapped angles.
    """
    return (err_deg + 180.0) % 360.0 - 180.0


def stats_summary(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"mean": np.nan, "rms": np.nan, "p2p": np.nan}
    mean = float(np.mean(x))
    rms = float(np.sqrt(np.mean((x - mean) ** 2)))
    p2p = float(np.max(x) - np.min(x))
    return {"mean": mean, "rms": rms, "p2p": p2p}


# =========================
# Main GUI
# =========================

class EncoderTestGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Encoder Test Platform (Professional Demo)")
        self.resize(1300, 820)

        # Theme-ish defaults for pyqtgraph
        pg.setConfigOptions(antialias=True)

        # Core configs
        self.cfg = DecoderConfig()
        self.sample_hz = 200.0
        self.dt = 1.0 / self.sample_hz

        # Devices (mock)
        self.motor = MockMotor()
        self.ref = MockEncoder(self.cfg.counts_per_rev, kind="ref")
        self.dut = MockEncoder(self.cfg.counts_per_rev, kind="dut")

        # State
        self.connected = False
        self.running = False
        self.mode = "Degrees"         # plot x-axis mode
        self.analysis = "Auto"        # Auto / Noise / Error
        self.noise_speed_threshold = 1.0  # deg/s

        # Buffers
        self.buf_time = RingBuffer(5000)
        self.buf_ref_deg = RingBuffer(5000)
        self.buf_dut_deg = RingBuffer(5000)
        self.buf_err_deg = RingBuffer(5000)
        self.buf_speed = RingBuffer(5000)

        # UI build
        self._build_ui()

        # v __init__ po _build_timer():
        self.acq_timer = QtCore.QTimer(self)
        self.acq_timer.timeout.connect(self._acq_tick)

        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self._plot_tick)


        self._log("GUI ready (Mock devices). Connect to start.")

    # ---------------- UI skeleton ----------------

    def _build_ui(self):
        # Central widget: split left (controls) and right (plots)
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left control panel
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(8)
        root.addLayout(left, 0)

        # Right plot panel
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(8)
        root.addLayout(right, 1)

        # ---- Group: Connection ----
        self.grp_conn = QtWidgets.QGroupBox("Connection")
        gl = QtWidgets.QGridLayout(self.grp_conn)
        gl.setColumnStretch(1, 1)

        self.cmb_port = QtWidgets.QComboBox()
        self.cmb_port.addItems(["MOCK_COM1", "MOCK_COM2"])

        self.btn_connect = QtWidgets.QPushButton("Connect")
        self.btn_connect.clicked.connect(self.on_connect)

        self.led_status = QtWidgets.QLabel("● DISCONNECTED")
        self.led_status.setStyleSheet("color: #C62828; font-weight: 600;")

        gl.addWidget(QtWidgets.QLabel("Port:"), 0, 0)
        gl.addWidget(self.cmb_port, 0, 1)
        gl.addWidget(self.btn_connect, 0, 2)
        gl.addWidget(self.led_status, 1, 0, 1, 3)

        left.addWidget(self.grp_conn)

        # ---- Group: Decoder settings ----
        self.grp_decoder = QtWidgets.QGroupBox("Decoder / Units")
        gd = QtWidgets.QGridLayout(self.grp_decoder)
        gd.setColumnStretch(1, 1)

        self.spin_st_bits = QtWidgets.QSpinBox()
        self.spin_st_bits.setRange(1, 31)
        self.spin_st_bits.setValue(self.cfg.singleturn_bits)

        self.spin_mt_bits = QtWidgets.QSpinBox()
        self.spin_mt_bits.setRange(0, 31)
        self.spin_mt_bits.setValue(self.cfg.multiturn_bits)

        self.spin_status_bits = QtWidgets.QSpinBox()
        self.spin_status_bits.setRange(0, 8)
        self.spin_status_bits.setValue(self.cfg.status_bits)

        self.spin_crc_bits = QtWidgets.QSpinBox()
        self.spin_crc_bits.setRange(0, 16)
        self.spin_crc_bits.setValue(self.cfg.crc_bits)

        self.spin_cpr = QtWidgets.QSpinBox()
        self.spin_cpr.setRange(1, 10_000_000)
        self.spin_cpr.setValue(self.cfg.counts_per_rev)

        self.chk_wrap = QtWidgets.QCheckBox("Wrap to 0–360° (singleturn)")
        self.chk_wrap.setChecked(True)

        self.btn_apply_decoder = QtWidgets.QPushButton("Apply")
        self.btn_apply_decoder.clicked.connect(self.on_apply_decoder)

        gd.addWidget(QtWidgets.QLabel("Singleturn bits:"), 0, 0)
        gd.addWidget(self.spin_st_bits, 0, 1)
        gd.addWidget(QtWidgets.QLabel("Multiturn bits:"), 1, 0)
        gd.addWidget(self.spin_mt_bits, 1, 1)
        gd.addWidget(QtWidgets.QLabel("Status bits:"), 2, 0)
        gd.addWidget(self.spin_status_bits, 2, 1)
        gd.addWidget(QtWidgets.QLabel("CRC bits:"), 3, 0)
        gd.addWidget(self.spin_crc_bits, 3, 1)
        gd.addWidget(QtWidgets.QLabel("Counts / rev:"), 4, 0)
        gd.addWidget(self.spin_cpr, 4, 1)
        gd.addWidget(self.chk_wrap, 5, 0, 1, 2)
        gd.addWidget(self.btn_apply_decoder, 6, 0, 1, 2)

        left.addWidget(self.grp_decoder)

        # ---- Group: Motor control (placeholder) ----
        self.grp_motor = QtWidgets.QGroupBox("Motor (Mock)")
        gm = QtWidgets.QGridLayout(self.grp_motor)
        gm.setColumnStretch(1, 1)

        self.chk_motor_enable = QtWidgets.QCheckBox("Enable")
        self.chk_motor_enable.setChecked(True)
        self.chk_motor_enable.stateChanged.connect(self.on_motor_enable)

        self.spin_speed = QtWidgets.QDoubleSpinBox()
        self.spin_speed.setRange(0.0, 2000.0)
        self.spin_speed.setDecimals(1)
        self.spin_speed.setSingleStep(5.0)
        self.spin_speed.setValue(30.0)
        self.spin_speed.valueChanged.connect(self.on_motor_speed)

        self.lbl_speed_live = QtWidgets.QLabel("0.0 deg/s")
        self.lbl_speed_live.setStyleSheet("font-weight: 600;")

        gm.addWidget(QtWidgets.QLabel("Speed:"), 0, 0)
        gm.addWidget(self.spin_speed, 0, 1)
        gm.addWidget(QtWidgets.QLabel("deg/s"), 0, 2)
        gm.addWidget(self.chk_motor_enable, 1, 0, 1, 2)
        gm.addWidget(QtWidgets.QLabel("Live:"), 2, 0)
        gm.addWidget(self.lbl_speed_live, 2, 1, 1, 2)

        left.addWidget(self.grp_motor)

        # ---- Group: Run control ----
        self.grp_run = QtWidgets.QGroupBox("Run Control")
        gr = QtWidgets.QGridLayout(self.grp_run)
        gr.setColumnStretch(1, 1)

        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.clicked.connect(self.on_start)
        self.btn_start.setEnabled(False)

        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)

        self.btn_clear = QtWidgets.QPushButton("Clear buffers")
        self.btn_clear.clicked.connect(self.on_clear)

        self.spin_rate = QtWidgets.QSpinBox()
        self.spin_rate.setRange(10, 2000)
        self.spin_rate.setValue(int(self.sample_hz))
        self.spin_rate.valueChanged.connect(self.on_rate_change)

        self.spin_noise_thr = QtWidgets.QDoubleSpinBox()
        self.spin_noise_thr.setRange(0.0, 2000.0)
        self.spin_noise_thr.setDecimals(1)
        self.spin_noise_thr.setValue(self.noise_speed_threshold)
        self.spin_noise_thr.valueChanged.connect(self.on_noise_thr_change)

        self.cmb_analysis = QtWidgets.QComboBox()
        self.cmb_analysis.addItems(["Auto", "Noise", "Error"])
        self.cmb_analysis.currentTextChanged.connect(self.on_analysis_changed)

        self.cmb_units = QtWidgets.QComboBox()
        self.cmb_units.addItems(["Degrees", "Counts"])
        self.cmb_units.currentTextChanged.connect(self.on_units_changed)

        gr.addWidget(QtWidgets.QLabel("Sampling (Hz):"), 0, 0)
        gr.addWidget(self.spin_rate, 0, 1)
        gr.addWidget(self.btn_start, 1, 0)
        gr.addWidget(self.btn_stop, 1, 1)
        gr.addWidget(self.btn_clear, 2, 0, 1, 2)
        gr.addWidget(QtWidgets.QLabel("Analysis:"), 3, 0)
        gr.addWidget(self.cmb_analysis, 3, 1)
        gr.addWidget(QtWidgets.QLabel("Auto noise thr (deg/s):"), 4, 0)
        gr.addWidget(self.spin_noise_thr, 4, 1)
        gr.addWidget(QtWidgets.QLabel("Plot units:"), 5, 0)
        gr.addWidget(self.cmb_units, 5, 1)

        left.addWidget(self.grp_run)

        # ---- Group: Live values + metrics ----
        self.grp_live = QtWidgets.QGroupBox("Live Values")
        gv = QtWidgets.QGridLayout(self.grp_live)
        gv.setColumnStretch(1, 1)

        self.lbl_ref_counts = QtWidgets.QLabel("—")
        self.lbl_dut_counts = QtWidgets.QLabel("—")
        self.lbl_ref_deg = QtWidgets.QLabel("—")
        self.lbl_dut_deg = QtWidgets.QLabel("—")
        for lab in [self.lbl_ref_counts, self.lbl_dut_counts, self.lbl_ref_deg, self.lbl_dut_deg]:
            lab.setStyleSheet("font-family: Consolas, monospace; font-weight: 600;")

        self.lbl_metric_mean = QtWidgets.QLabel("mean: —")
        self.lbl_metric_rms = QtWidgets.QLabel("rms: —")
        self.lbl_metric_p2p = QtWidgets.QLabel("p2p: —")
        for lab in [self.lbl_metric_mean, self.lbl_metric_rms, self.lbl_metric_p2p]:
            lab.setStyleSheet("font-family: Consolas, monospace;")

        gv.addWidget(QtWidgets.QLabel("REF counts:"), 0, 0)
        gv.addWidget(self.lbl_ref_counts, 0, 1)
        gv.addWidget(QtWidgets.QLabel("DUT counts:"), 1, 0)
        gv.addWidget(self.lbl_dut_counts, 1, 1)
        gv.addWidget(QtWidgets.QLabel("REF deg:"), 2, 0)
        gv.addWidget(self.lbl_ref_deg, 2, 1)
        gv.addWidget(QtWidgets.QLabel("DUT deg:"), 3, 0)
        gv.addWidget(self.lbl_dut_deg, 3, 1)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        gv.addWidget(sep, 4, 0, 1, 2)

        gv.addWidget(QtWidgets.QLabel("Metric (analysis):"), 5, 0, 1, 2)
        gv.addWidget(self.lbl_metric_mean, 6, 0, 1, 2)
        gv.addWidget(self.lbl_metric_rms, 7, 0, 1, 2)
        gv.addWidget(self.lbl_metric_p2p, 8, 0, 1, 2)

        left.addWidget(self.grp_live)

        left.addStretch(1)

        # ---- LOG dock ----
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setFont(QtGui.QFont("Consolas", 9))

        dock = QtWidgets.QDockWidget("Log", self)
        dock.setWidget(self.log)
        dock.setAllowedAreas(QtCore.Qt.BottomDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

        # ---- Right: plots ----
        self.plot_pos = pg.PlotWidget(title="Position (REF vs DUT)")
        self.plot_pos.showGrid(x=True, y=True, alpha=0.3)
        self.curve_ref = self.plot_pos.plot(pen=pg.mkPen(width=2), name="REF")
        self.curve_dut = self.plot_pos.plot(pen=pg.mkPen(width=2, style=QtCore.Qt.DashLine), name="DUT")

        self.plot_ana = pg.PlotWidget(title="Analysis (Noise or Error)")
        self.plot_ana.showGrid(x=True, y=True, alpha=0.3)
        self.curve_ana = self.plot_ana.plot(pen=pg.mkPen(width=2))

        right.addWidget(self.plot_pos, 2)
        right.addWidget(self.plot_ana, 1)

        # ---- Status bar ----
        self.status = self.statusBar()
        self.lbl_status_main = QtWidgets.QLabel("Disconnected")
        self.lbl_status_rate = QtWidgets.QLabel(f"{int(self.sample_hz)} Hz")
        self.lbl_status_mode = QtWidgets.QLabel("Analysis: Auto")
        self.status.addWidget(self.lbl_status_main, 1)
        self.status.addPermanentWidget(self.lbl_status_rate)
        self.status.addPermanentWidget(self.lbl_status_mode)

        # Initial disable
        self._set_controls_enabled(connected=False)


    # ---------------- Actions ----------------

    def _set_controls_enabled(self, connected: bool):
        self.grp_decoder.setEnabled(connected)
        self.grp_motor.setEnabled(connected)
        self.grp_run.setEnabled(connected)
        self.btn_start.setEnabled(connected and not self.running)
        self.btn_stop.setEnabled(connected and self.running)

    def _apply_timer_rate(self):
        hz = max(10.0, float(self.sample_hz))
        ms = int(round(1000.0 / hz))
        self.lbl_status_rate.setText(f"{int(hz)} Hz")

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")

    def on_connect(self):
        if not self.connected:
            port = self.cmb_port.currentText()
            self.connected = True
            self.btn_connect.setText("Disconnect")
            self.led_status.setText("● CONNECTED")
            self.led_status.setStyleSheet("color: #2E7D32; font-weight: 600;")
            self.lbl_status_main.setText(f"Connected: {port} (MOCK)")
            self._set_controls_enabled(connected=True)
            self._log(f"Connected to {port} (mock).")
        else:
            self.on_stop()
            self.connected = False
            self.btn_connect.setText("Connect")
            self.led_status.setText("● DISCONNECTED")
            self.led_status.setStyleSheet("color: #C62828; font-weight: 600;")
            self.lbl_status_main.setText("Disconnected")
            self._set_controls_enabled(connected=False)
            self._log("Disconnected.")

    def on_apply_decoder(self):
        self.cfg.singleturn_bits = int(self.spin_st_bits.value())
        self.cfg.multiturn_bits = int(self.spin_mt_bits.value())
        self.cfg.status_bits = int(self.spin_status_bits.value())
        self.cfg.crc_bits = int(self.spin_crc_bits.value())
        self.cfg.counts_per_rev = int(self.spin_cpr.value())

        # re-init mocks to reflect CPR (demo)
        self.ref = MockEncoder(self.cfg.counts_per_rev, kind="ref")
        self.dut = MockEncoder(self.cfg.counts_per_rev, kind="dut")

        self._log(
            f"Decoder applied: ST={self.cfg.singleturn_bits} MT={self.cfg.multiturn_bits} "
            f"STATUS={self.cfg.status_bits} CRC={self.cfg.crc_bits} CPR={self.cfg.counts_per_rev}"
        )

    def on_motor_speed(self, v):
        self.motor.set_speed(float(v))

    def on_motor_enable(self, state):
        self.motor.set_enabled(state == QtCore.Qt.Checked)

    def on_rate_change(self, v):
        self.sample_hz = float(v)
        self._apply_timer_rate()

    def on_noise_thr_change(self, v):
        self.noise_speed_threshold = float(v)

    def on_analysis_changed(self, text):
        self.analysis = text
        self.lbl_status_mode.setText(f"Analysis: {text}")

    def on_units_changed(self, text):
        self.mode = text

    def on_start(self):
        if not self.connected:
            return
        self.running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self._apply_timer_rate()  # acq interval
        self.acq_timer.start()
        self.plot_timer.start()

        self._log("Acquisition started.")

    def on_stop(self):
        if not self.running:
            return
        self.running = False
        self.acq_timer.stop()
        self.plot_timer.stop()

        self.btn_start.setEnabled(self.connected)
        self.btn_stop.setEnabled(False)
        self._log("Acquisition stopped.")

    def on_clear(self):
        self.buf_time.clear()
        self.buf_ref_deg.clear()
        self.buf_dut_deg.clear()
        self.buf_err_deg.clear()
        self.buf_speed.clear()
        self._update_plots()
        self._update_metrics(np.array([]))
        self._log("Buffers cleared.")

    # ---------------- Core loop ----------------

    def _acq_tick(self):
        # samo bere, računa minimalno in append-a v buffer
        angle_deg, speed_deg_s = self.motor.step()

        ref_counts = self.ref.read_counts_from_angle(angle_deg)
        dut_counts = self.dut.read_counts_from_angle(angle_deg)

        if self.chk_wrap.isChecked():
            ref_counts = self.cfg.wrap_counts(ref_counts)
            dut_counts = self.cfg.wrap_counts(dut_counts)

        ref_deg = self.cfg.counts_to_deg(ref_counts)
        dut_deg = self.cfg.counts_to_deg(dut_counts)
        err_deg = wrap_error_deg(np.array([dut_deg - ref_deg]))[0]

        t = time.perf_counter()
        self.buf_time.append(t)
        self.buf_ref_deg.append(ref_deg)
        self.buf_dut_deg.append(dut_deg)
        self.buf_err_deg.append(err_deg)
        self.buf_speed.append(speed_deg_s)

        # live labels (to je OK)
        self.lbl_ref_counts.setText(str(ref_counts))
        self.lbl_dut_counts.setText(str(dut_counts))
        self.lbl_ref_deg.setText(f"{ref_deg:8.3f}°")
        self.lbl_dut_deg.setText(f"{dut_deg:8.3f}°")
        self.lbl_speed_live.setText(f"{speed_deg_s:0.1f} deg/s")

    def _plot_tick(self):
        # samo plot + metrics
        self._update_plots()

        ana = self._compute_analysis_series()  # samo 1x
        self._update_metrics(ana)

    def _compute_analysis_series(self) -> np.ndarray:
        speed = self.buf_speed.array()
        err = self.buf_err_deg.array()

        if err.size == 0:
            return err

        if self.analysis == "Noise":
            # Noise is error-like but when stationary, you may prefer DUT - mean(DUT).
            # Here we use error signal and detrend it.
            x = err - np.mean(err)
            return x

        if self.analysis == "Error":
            return err

        # Auto: decide by speed
        v = float(np.mean(np.abs(speed[-200:]))) if speed.size else 0.0
        if v < self.noise_speed_threshold:
            x = err - np.mean(err)
            return x
        return err

    def _update_plots(self):
        t = self.buf_time.array()
        ref_deg = self.buf_ref_deg.array()
        dut_deg = self.buf_dut_deg.array()

        if t.size == 0:
            self.curve_ref.setData([])
            self.curve_dut.setData([])
            self.curve_ana.setData([])
            return

        # use relative time axis for stability
        tt = t - t[0]

        # Position plot: degrees or counts
        if self.mode == "Degrees":
            y_ref = ref_deg
            y_dut = dut_deg
            self.plot_pos.setLabel("left", "deg")
        else:
            # back-convert degrees to counts approx for demo
            cpr = max(1, self.cfg.counts_per_rev)
            y_ref = (ref_deg / 360.0) * cpr
            y_dut = (dut_deg / 360.0) * cpr
            self.plot_pos.setLabel("left", "counts")

        self.curve_ref.setData(tt, y_ref)
        self.curve_dut.setData(tt, y_dut)

        # Analysis plot always in degrees for demo
        ana = self._compute_analysis_series()
        if ana.size:
            self.curve_ana.setData(tt[-ana.size:], ana)
        else:
            self.curve_ana.setData([])

        # Update titles to reflect mode
        self.plot_pos.setTitle(f"Position (REF vs DUT) — Units: {self.mode}")
        self.plot_ana.setTitle(f"Analysis — {self.analysis} (Auto thr={self.noise_speed_threshold:.1f} deg/s)")

    def _update_metrics(self, ana: np.ndarray):
        s = stats_summary(ana)
        self.lbl_metric_mean.setText(f"mean: {s['mean']:+.4f} deg")
        self.lbl_metric_rms.setText(f"rms : {s['rms']:+.4f} deg")
        self.lbl_metric_p2p.setText(f"p2p : {s['p2p']:+.4f} deg")


# =========================
# Entrypoint
# =========================

def main():
    app = QtWidgets.QApplication(sys.argv)

    # Optional: nicer fonts on Windows
    app.setFont(QtGui.QFont("Segoe UI", 9))

    w = EncoderTestGUI()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()