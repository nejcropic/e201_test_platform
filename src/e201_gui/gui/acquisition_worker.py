import time
import traceback

from queue import Queue, Empty
from PyQt5.QtCore import QThread, pyqtSignal
from e201_gui.e201_drivers.master import Master


class AcquisitionWorker(QThread):
    """
    Worker that continuously samples two E201 devices (e.g. Q + B).

    Emits synchronized samples.
    If device is not connected or read fails -> value is None.
    """

    error_signal = pyqtSignal(object)
    finished_signal = pyqtSignal()
    register_response_signal = pyqtSignal(object)
    error_response_signal = pyqtSignal(object)

    def __init__(self, buffer, parser):
        super().__init__()
        self.buffer = buffer
        self.master = Master(
            {
                "e201_dut": {
                    "type": "E2019P",
                    "comport": None,
                },
                "e201_ref": {
                    "type": "E2019Q",
                    "comport": None,
                },
            }
        )
        self.parser = parser
        self.last_position = None
        self.command_queue = Queue()
        self.running = True
        self.sample_index = 0

        # processing state
        self.prev_err_deg = None
        self.prev_dut_counts = None
        self.prev_ref_counts = None
        self.set_zero_offset = False
        self._error_offset = 0.0
        self.recording = False
        self.recorded_data = []

        self.invert_dut = False
        self.noise_source = "DUT"  # or "REF"

    def run(self):
        try:
            while self.running:
                t0 = time.perf_counter()
                self._loop_iteration()
                dt = time.perf_counter() - t0
                sleep_time = max(0.0, 0.001 - dt)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as e:
            self.error_signal.emit([e, traceback.format_exc()])

    def _loop_iteration(self):
        self._handle_commands()
        if self.master.dut is None and self.master.ref is None:
            time.sleep(0.01)
            return

        self._read_encoders()

    def _read_encoders(self):
        try:
            frame = self.master.read_position()
            parsed = self.parser.parse_position(frame)

            processed = self._compute_processed_sample(parsed)

            self.buffer.append(
                sample_idx=processed["sample_idx"],
                ts=processed["ts"],
                dut_counts=processed["dut_counts"],
                ref_counts=processed["ref_counts"],
                dut_deg=processed["dut_deg"],
                ref_deg=processed["ref_deg"],
                err_deg=processed["err_deg"],
                inl_deg=processed["inl_deg"],
                dnl_deg=processed["dnl_deg"],
                noise=processed["noise"],
                multiturn=processed["multiturn"],
            )

            if self.recording:
                self.recorded_data.append(
                    {
                        "x": processed["sample_idx"],
                        "ref_counts": processed["ref_counts"],
                        "dut_counts": processed["dut_counts"],
                        "ref_deg": processed["ref_deg"],
                        "dut_deg": processed["dut_deg"],
                    }
                )

            self.last_position = parsed.copy()
            self.latest_sample = processed
            self.sample_index += 1

        except Exception as e:
            import traceback

            traceback.print_exc()
            self.error_signal.emit(str(e))

    @staticmethod
    def _wrap_error_deg(err):
        return (err + 180.0) % 360.0 - 180.0

    def _compute_processed_sample(self, parsed: dict) -> dict:
        dut_counts = int(parsed["Position"])
        ref_counts = int(parsed["Reference"])
        ts = float(parsed["Timer"])

        if self.invert_dut:
            dut_counts = (-dut_counts) % self.parser.dut_settings["resolution"]

        dut_deg = dut_counts * self.parser.dut_resolution
        ref_deg = (ref_counts * self.parser.ref_resolution) % 360.0

        err_deg = self._wrap_error_deg(dut_deg - ref_deg) - self._error_offset
        inl_deg = err_deg

        if self.prev_err_deg is None:
            dnl_deg = 0.0
        else:
            dnl_deg = err_deg - self.prev_err_deg

        if self.noise_source == "DUT":
            if self.prev_dut_counts is None:
                noise = 0.0
            else:
                noise = float(dut_counts - self.prev_dut_counts)
        else:
            if self.prev_ref_counts is None:
                noise = 0.0
            else:
                noise = float(ref_counts - self.prev_ref_counts)

        if self.set_zero_offset:
            self._error_offset = err_deg
            self.set_zero_offset = False
        self.prev_err_deg = err_deg
        self.prev_dut_counts = dut_counts
        self.prev_ref_counts = ref_counts

        return {
            "sample_idx": self.sample_index,
            "ts": ts,
            "dut_counts": dut_counts,
            "ref_counts": ref_counts,
            "dut_deg": dut_deg,
            "ref_deg": ref_deg,
            "err_deg": err_deg,
            "inl_deg": inl_deg,
            "dnl_deg": dnl_deg,
            "noise": noise,
            "status": parsed.get("Status", -1),
            "multiturn": parsed.get("Multiturn", 0),
        }

    def enqueue_command(self, command: str, *args):
        self.command_queue.put((command, args))

    def _handle_commands(self):
        try:
            command, args = self.command_queue.get_nowait()
            command = getattr(self, command)
            response = command(*args)
            if response is not None:
                self.register_response_signal.emit(
                    {
                        "response": response,
                    }
                )
        except Empty:
            pass

    def connect_dut(self, e201_type, port):
        self.master.dut = self.master._init_device(
            device_cfg={
                "type": e201_type,
                "comport": port,
            },
            role="DUT",
        )
        if self.master.ref is not None:
            self.master.enable_synced_sampling()
        else:
            self.master.disable_synced_sampling()

    def connect_ref(self, e201_type, port):
        self.master.ref = self.master._init_device(
            device_cfg={
                "type": e201_type,
                "comport": port,
            },
            role="REF",
        )
        if self.master.dut is not None:
            self.master.enable_synced_sampling()
        else:
            self.master.disable_synced_sampling()

    def set_dut_communication(self, config):
        self.master.initialize_device(config)

    def enable_synced_sampling(self):
        self.master.enable_synced_sampling()

    def disable_synced_sampling(self):
        self.master.disable_synced_sampling()

    def read_dut_register(self, bank: int, address: int | str, length: int, signed):
        if self.master.dut is None:
            self.error_response_signal.emit("DUT not connected!")

        try:
            resp = self.master.read_registers(bank, address, length, signed)
            self.register_response_signal.emit(resp)
        except Exception as e:
            self.error_response_signal.emit(e)

    def write_dut_register(self, value: int, bank: int, address: int | str, length: int, signed):
        if self.master.dut is None:
            self.error_response_signal.emit("DUT not connected!")

        try:
            resp = self.master.write_registers(value, bank, address, length, signed)
            return resp
        except Exception as e:
            self.error_response_signal.emit(e)

    def set_multiturn(self, mt_value: int):
        if self.master.dut is None:
            self.error_response_signal.emit("DUT not connected!")

        try:
            resp = self.master.set_multiturn(mt_value)
            return resp
        except Exception as e:
            self.error_response_signal.emit(e)

    def set_position_offset(self, offset_value: int):
        if self.master.dut is None:
            self.error_response_signal.emit("DUT not connected!")

        try:
            resp = self.master.set_position_offset(offset_value)
            return resp
        except Exception as e:
            self.error_response_signal.emit(e)

    def set_register_access(self, preset_values: dict):
        self.master.set_register_access(preset_values)

    def dut_power_on(self, voltage):
        if self.master.dut is not None:
            return

        self.master.dut.power_on(voltage)

    def ref_power_on(self, voltage):
        if self.master.ref is not None:
            return

        self.master.ref.power_on(voltage)

    def dut_power_off(self):
        if self.master.dut is not None:
            return

        self.master.dut.power_off()

    def ref_power_off(self):
        if self.master.ref is not None:
            return

        self.master.ref.power_off()

    def dut_power_cycle(self, voltage):
        if self.master.dut is not None:
            return

        self.master.dut.power_off()
        self.master.dut.power_on(voltage)

    def ref_power_cycle(self, voltage):
        if self.master.ref is not None:
            return

        self.master.ref.power_off()
        self.master.ref.power_on(voltage)

    def close_dut(self):
        if self.master.dut.trigger_enabled:
            self.master.disable_synced_sampling()

        self.master.close_dut()

    def close_ref(self):
        if self.master.ref.trigger_enabled:
            self.master.disable_synced_sampling()

        self.master.close_ref()

    def stop_worker(self):
        self.running = False

    def disconnect(self):
        self.master.close_connection()
