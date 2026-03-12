import time
import traceback
from typing import Any, Dict
from e201_test_platform.e201_versions.e201b import E201B
from e201_test_platform.e201_versions.e201q import E201Q

from queue import Queue, Empty
from PyQt5.QtCore import QThread, pyqtSignal

supported_e201 = {
    "E201B": E201B,
    "E201Q": E201Q
}


class AcquisitionWorker(QThread):
    """
    Worker that continuously samples two E201 devices (e.g. Q + B).

    Emits synchronized samples.
    If device is not connected or read fails -> value is None.
    """

    sample_signal = pyqtSignal(object)  # dict with timestamp + values
    error_signal = pyqtSignal(object)
    finished_signal = pyqtSignal()
    register_response_signal = pyqtSignal(object)
    detailed_status_signal = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.dut: E201B | None = None
        self.ref: E201Q | None = None

        self.sync_reading = False

        self.loop_flag = True
        self.command_queue = Queue()

    def run(self):
        try:
            while self.loop_flag:
                self._loop_iteration()


        except Exception as e:
            self.error_signal.emit([e, traceback.format_exc()])
        finally:
            self.finished_signal.emit()

    def _loop_iteration(self):
        self._handle_commands()
        if self.dut is None and self.ref is None:
            time.sleep(0.01)
            return

        self.sample_signal.emit(self._read_encoders())

    def _read_encoders(self) -> tuple:
        dut_position = None
        ref_position = None

        t = time.perf_counter()
        try:
            if self.sync_reading and self.dut is not None and self.ref is not None:
                dut_raw = self.dut.read_position()
                ref_raw = self.ref.read()
                dut_position = self.dut.parse_position(dut_raw)
                ref_position = self.ref.parse_slave_position(ref_raw)

            else:
                if self.dut is not None:
                    if self.dut.initialized:
                        dut_raw = self.dut.read_position()
                        dut_position = self.dut.parse_position(dut_raw)

                if self.ref is not None:
                    if self.ref.initialized:
                        ref_raw = self.ref.read_position()
                        ref_position = self.ref.parse_position(ref_raw)

        except Exception as e:
            self.error_signal.emit(str(e))

        return t, dut_position, ref_position

    def enqueue_command(self, command: str, *args):
        self.command_queue.put((command, args))

    def _handle_commands(self):
        try:
            command, args = self.command_queue.get_nowait()
            result = getattr(self, command)(*args)

            if result is not None:
                self.register_response_signal.emit({
                    "command": command,
                    "result": result
                })
        except Empty:
            pass

    def initialize_dut(self, e201_type, port, parameters):
        dut_instance = supported_e201[e201_type]
        self.dut = dut_instance(port, parameters)
        self.dut.initialize()
        self.sync_reading = False

    def initialize_ref(self, e201_type, port, parameters):
        ref_instance = supported_e201[e201_type]
        self.ref = ref_instance(port, parameters)
        self.ref.initialize()
        self.sync_reading = False

    def close_dut(self):
        self.dut.close()
        self.dut = None

    def close_ref(self):
        self.ref.close()
        self.ref = None

    def initialize_encoders(self, e201_ref: E201Q | None = None, e201_dut: E201B | None = None):
        """
        Pass already constructed encoder objects.
        """
        self.ref = e201_ref
        self.dut = e201_dut

    def enable_synced_sampling(self):
        if self.dut is None or self.ref is None:
            print("Connection with DUT and reference have to be established to sync samples!")
            return

        self.ref.enable_trigger_slave()
        self.dut.enable_trigger_master()
        self.sync_reading = True

    def disable_synced_sampling(self):
        if self.dut is None or self.ref is None:
            print("Connection with DUT and reference have to be established to sync samples!")
            return

        self.ref.disable_trigger()
        self.dut.disable_trigger()
        self.sync_reading = False

    def read_dut_register(self, bank: int, address: int | str, length: int, signed):
        if self.dut is None:
            return {"error": "DUT not connected"}

        try:
            resp = self.dut.read_registers(bank, address, length, signed)
            return resp
        except Exception as e:
            return {"error": str(e)}

    def write_dut_register(self, value: int, bank: int, address: int | str, length: int, signed):
        if self.dut is None:
            return {"error": "DUT not connected"}

        try:
            resp = self.dut.write_registers(value, bank, address, length, signed)
            return {"status": "ok"}
        except Exception as e:
            return {"error": str(e)}

    def write_dut_register_param(self, value, param: str):
        if self.dut is None:
            return {"error": "DUT not connected"}

        try:
            resp = self.dut.write_registers_params(value, param)
            return {"raw": resp}
        except Exception as e:
            return {"error": str(e)}

    def read_dut_register_param(self, param: str):
        if self.dut is None:
            return {"error": "DUT not connected"}

        try:
            resp = self.dut.read_registers_params(param)
            return {"raw": resp}
        except Exception as e:
            return {"error": str(e)}

    def set_multiturn(self, value: int):
        if self.dut is None:
            return {"error": "DUT not connected"}

        try:
            resp = self.dut.set_multiturn(value)
            return {"raw": resp}
        except Exception as e:
            return {"error": str(e)}

    def read_detailed_status(self):
        if self.dut is None:
            return {"error": "DUT not connected"}

        try:
            resp = self.dut.read_detailed_status()
            return {"raw": resp}
        except Exception as e:
            return {"error": str(e)}

    def stop_worker(self):
        self.loop_flag = False
        self.wait()

    def disconnect(self):
        if self.ref:
            try:
                self.ref.close()
            except Exception:
                pass

        if self.dut:
            try:
                self.dut.close()
            except Exception:
                pass
