import time
from typing import Optional, List
from dataclasses import dataclass

import serial.tools.list_ports  # type: ignore #import serial module
from serial.serialutil import SerialException  # type: ignore #import serial module

from letp_e2019.serial_api.p91 import P911, InterfaceNotFoundException
from letp_e2019.e2019_helpers import e2019p_errors, e2019b_errors
from letp_e2019.serial_api.serial_port import DeviceNotFound


class E2019(P911):
    NAME: str = "E201"
    type = None
    available_freq: dict = {}

    def __init__(self, com):
        try:
            super().__init__(com)
        except DeviceNotFound:
            devices = [device.e2019_comport for device in get_all_e201()]
            raise ConnectionError(f"E201 comports found: {devices}")

        self.trigger_enabled = False
        self.read_command = None
        self.bytes = None

    def read_position(self) -> str:
        return self._serial_port.execute_command_with_response(self.read_command)

    def enable_trigger_master(self):
        self.trigger_enabled = True
        return self._serial_port.execute_command_with_response("TM")

    def enable_trigger_slave(self):
        self.trigger_enabled = True
        return self._serial_port.execute_command_with_response("TS")

    def set_trigger_data_format(self, data_format: int):
        return self._serial_port.execute_command_with_response(f"T{data_format}")

    def disable_trigger(self):
        self.trigger_enabled = False
        return self._serial_port.execute_command_with_response("Tx")

    def generate_trigger_pulse(self):
        if self.trigger_enabled:
            return self._serial_port.execute_command_with_response("TT")

    def set_read_command(self, communication: str):
        pass

    def read_clock_frequency(self):
        return self.execute_command_with_response("m")

    def set_clock_frequency(self, freq_khz: int):
        if freq_khz not in self.available_freq:
            raise ValueError(f"Frequency unavailable! Available frequencies [kHz]: {list(self.available_freq.keys())}")

        if self.type == "B":
            self.execute_command_with_response(f"M{self.available_freq[freq_khz]:02d}")
        else:
            self.execute_command_with_response(f"M{self.available_freq[freq_khz]}")

        freq_set = self.read_clock_frequency().split(" ")[0]
        if freq_khz != int(freq_set):
            if int(freq_set) != self.available_freq[int(freq_khz)]:
                raise ValueError("Frequency not set!")

    @staticmethod
    def _parse_register_response(resp: str) -> dict:
        response = resp.strip().split(":")
        return {"status": response[0], "details": response[1], "data_hex": response[2]}

    def _raise_on_status(self, status: str) -> None:
        status_maps = {
            "P": e2019p_errors,
            "B": e2019b_errors,
            None: None,
        }

        status_helper = status_maps.get(self.type)
        if status_helper is None:
            return

        try:
            response = status_helper[status]
        except KeyError:
            raise RuntimeError(f"Unknown status: {status}")

        if response != "OK":
            raise RegisterAccessStatusErrorException(status, response)


class E2019Power(E2019):
    def power_on(self, voltage_mv: Optional[int] = None) -> None:
        response = self._serial_port.execute_command_with_response("n")

        time.sleep(self.POWER_ON_DURATION_s)
        if "ON" not in response:
            raise Exception(f"power_on failed! Response: {response}")  # pylint: disable=broad-exception-raised

    def power_off(self) -> None:
        response = self._serial_port.execute_command_with_response("f")

        time.sleep(self.POWER_OFF_DURATION_s)
        if "OFF" not in response:
            raise Exception(f"Power off failed. Response: {response}")  # pylint: disable=broad-exception-raised


@dataclass
class E2019Info:  # pylint: disable=missing-class-docstring
    e2019_type: str
    e2019_comport: str


def get_all_e201() -> List[E2019Info]:
    """Get all connected e2019 devices"""
    devices = []

    for port in serial.tools.list_ports.comports():
        if port.vid == E2019.VID:
            try:
                e2019 = E2019(port.name)
                v = e2019.get_version().strip().split(" ")[0].replace("-", "")
                e2019.close()

                devices += [E2019Info(v, port.name)]

            except InterfaceNotFoundException:
                pass

            except SerialException:
                pass

    return devices


class RegisterAccessStatusErrorException(Exception):
    """Error accessing register"""

    def __init__(self, status: str, message: str):
        self.status = status
        self.message = message
        super().__init__(f"[STATUS {status}] {message}")
