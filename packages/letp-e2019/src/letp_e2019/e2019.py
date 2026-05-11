"""
COPYRIGHT(c) 2020 RLS d.o.o, Pod vrbami 2, 1218 Komenda, Slovenia

file:      e2019.py
brief:     Base class for E2019 device communication over UART protocol.
author(s): Nejc Ropič
date:      17.4.2026

details:   Provides serial communication, trigger control (master/slave), and position
           readout via configurable commands. Handles clock setup and status parsing.
"""

import time
from typing import Optional, List
from dataclasses import dataclass

import serial.tools.list_ports  # type: ignore # import serial module
from serial.serialutil import SerialException  # type: ignore # import serial module

from letp_e2019.serial_api.p91 import P911, InterfaceNotFoundException
from letp_e2019.e2019_helpers import e2019p_errors, e2019b_errors
from letp_e2019.serial_api.serial_port import DeviceNotFound


class E2019(P911):
    NAME: str = "E201"
    type = None
    available_freq: dict = {}

    def __init__(self, com):
        """Initialize E2019 device on given COM port."""
        try:
            super().__init__(com)
        except DeviceNotFound:
            devices = [device.e2019_comport for device in get_all_e201()]
            raise ConnectionError(f"E201 comports found: {devices}")

        self._trigger_enabled = False
        self.read_command = ""
        self.bytes = 4
        self._is_master = False

    def read_position(self) -> str:
        """
        Read position depending on trigger configuration.

        - Master:
            * Type B/P → send read command
            * Other types → generate trigger pulse
        - Slave (trigger enabled) → read buffered response
        - No trigger → standard command read
        """
        if self._trigger_enabled:
            if self._is_master:
                # B/P types do NOT use trigger pulse
                if self.type in ["P", "B"]:
                    return self._serial_port.execute_command_with_response(self.read_command)
                else:
                    return self.generate_trigger_pulse()
            else:
                # Slave: read buffered response
                return self.read()

        # Normal (non-trigger) read
        return self._serial_port.execute_command_with_response(self.read_command)

    def enable_trigger_master(self):
        """Enable trigger mode as master (generates trigger pulses)."""
        self._trigger_enabled = True
        self._is_master = True
        return self._serial_port.execute_command_with_response("TM")

    def enable_trigger_slave(self):
        """Enable trigger mode as slave (responds to external trigger)."""
        self._trigger_enabled = True
        self._is_master = False
        return self._serial_port.execute_command_with_response("TS")

    def set_trigger_data_format(self, data_format: int):
        """Set trigger data format."""
        return self._serial_port.execute_command_with_response(f"T{data_format}")

    def disable_trigger(self):
        """Disable trigger mode."""
        self._trigger_enabled = False
        return self._serial_port.execute_command_with_response("Tx")

    def generate_trigger_pulse(self):
        """Generate trigger pulse (master only)."""
        if not self._trigger_enabled:
            raise ValueError("Trigger not enabled!")

        return self._serial_port.execute_command_with_response("TT")

    def set_read_command(self, communication: str):
        """Set read command based on communication type."""
        pass

    def read_clock_frequency(self):
        """Read current clock frequency."""
        return self.execute_command_with_response("m")

    def set_clock_frequency(self, freq_khz: int):
        """
        Set clock frequency.

        Args:
            freq_khz: Desired frequency in kHz.

        Raises:
            ValueError: If frequency is unavailable or not set correctly.
        """
        if freq_khz not in self.available_freq:
            raise ValueError(f"Frequency unavailable! Available frequencies [kHz]: {list(self.available_freq.keys())}")

        if self.type == "B":
            self.execute_command_with_response(f"M{self.available_freq[freq_khz]:02d}")
        else:
            self.execute_command_with_response(f"M{self.available_freq[freq_khz]}")

        freq_set = self.read_clock_frequency().split(" ")[0]

        freq_check = freq_khz
        if self.type == "B":
            freq_check = self.available_freq[int(freq_khz)]

        if freq_check != int(freq_set):
            raise ValueError("Frequency not set!")

    @staticmethod
    def _parse_register_response(resp: str) -> dict:
        """
        Parse register response string.

        Args:
            resp: Raw response string.

        Returns:
            dict: Parsed response with status, details, and data_hex.
        """
        response = resp.strip().split(":")
        return {"status": response[0], "details": response[1], "data_hex": response[2]}

    def _raise_on_status(self, status: str) -> None:
        """
        Raise exception if status indicates error.

        Args:
            status: Status code from device.
        """
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

    def write_registers(self, value: int, bank: int, address: int | str, length: int, is_signed: bool = False):
        """
        Write value to consecutive registers.

        Args:
            value: Value to write.
            bank: Register bank.
            address: Start address.
            length: Number of bytes.
            is_signed: Interpret value as signed.
        """
        raise NotImplementedError(f"Register access not supported for E2019-{self.type}")

    def read_registers(self, bank: int, address: int | str, length: int, is_signed: bool = False):
        """
        Read multiple registers.

        Args:
            bank: Register bank.
            address: Start address.
            length: Number of bytes.
            is_signed: Unused (kept for API compatibility).

        Returns:
            Raw bytes read from device.
        """
        raise NotImplementedError(f"Register access not supported for E2019-{self.type}")


class E2019Power(E2019):
    def power_on(self, voltage_mv: Optional[int] = None) -> None:
        """Turn device power on."""
        response = self._serial_port.execute_command_with_response("n")

        time.sleep(self.POWER_ON_DURATION_s)
        if "ON" not in response:
            raise Exception(f"power_on failed! Response: {response}")

    def power_off(self) -> None:
        """Turn device power off."""
        response = self._serial_port.execute_command_with_response("f")

        time.sleep(self.POWER_OFF_DURATION_s)
        if "OFF" not in response:
            raise Exception(f"Power off failed. Response: {response}")


@dataclass
class E2019Info:  # pylint: disable=missing-class-docstring
    e2019_type: str
    e2019_comport: str


def get_all_e201() -> List[E2019Info]:
    """Return list of detected E2019 devices (type and COM port)."""
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
    """Exception raised when register access returns an error status."""

    def __init__(self, status: str, message: str):
        self.status = status
        self.message = message
        super().__init__(f"[STATUS {status}] {message}")
