"""
COPYRIGHT(c) 2020 RLS d.o.o, Pod vrbami 2, 1218 Komenda, Slovenia

file:      P91.py
brief:     Implementation of the P911 and P912 communication interface, as well as the EncoderDebugUnitP91 debug unit
author(s): Matjaž Muc
date:      31.5.2018

details:   Implementation of the P911 and P912 communication interface, as well as the EncoderDebugUnitP91 debug unit
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union


from letp_e2019.serial_api.serial_port import SerialPort


class P91(ABC):
    """Base P91<x> class; Implements common functionality."""

    POWER_ON_DURATION_s = 0.3
    POWER_OFF_DURATION_s = 0.1
    _serial_port: SerialPort

    def read(self) -> str:
        return self._serial_port.read()

    def write(self, command: Union[str, bytearray]) -> None:
        return self._serial_port.write(command)

    def execute_command_with_response(self, command: Union[str, bytearray]) -> str:
        return self._serial_port.execute_command_with_response(command)

    def get_version(self) -> str:
        return self._serial_port.execute_command_with_response("v")

    def get_build_number(self) -> str:
        return self._serial_port.execute_command_with_response("o")

    def get_supply(self) -> str:
        return self._serial_port.execute_command_with_response("e")

    def open(self) -> None:
        self._serial_port.open()

    def close(self) -> None:
        self._serial_port.close()

    def enter_debug_unit(self) -> None:
        """Initialize debug unit"""
        response = self._serial_port.execute_command_with_response("H")
        if response != "Entering into debug unit...":
            raise Exception(  # pylint: disable=broad-exception-raised
                f"Entering debug mode failed! Device returned: {response}"
            )

    @abstractmethod
    def power_off(self) -> None:
        """Attempt to power off the interface"""

    @abstractmethod
    def power_on(self, voltage_mv: Optional[int] = None) -> None:
        """Attempt to power on the interface"""


class P911(P91):  # pylint: disable=missing-class-docstring
    NAME: str = "P911"
    VID: int = 0x483

    def __init__(self, com: Optional[str] = None) -> None:
        if com is not None:
            self._serial_port = SerialPort(com)
        else:
            self._serial_port = SerialPort(self.VID)

        # authenticate device
        response = self._serial_port.execute_command_with_response("v")
        if str(self) not in response:
            self.close()
            raise InterfaceNotFoundException(f"Connect {str(self)} to the comm. port!")

    def __str__(self) -> str:
        return self.NAME

    def power_on(self, voltage_mv: Optional[int] = None) -> None:
        response = self._serial_port.execute_command_with_response("N")

        time.sleep(self.POWER_ON_DURATION_s)
        if "ON" not in response:
            raise Exception(f"power_on failed! Response: {response}")  # pylint: disable=broad-exception-raised

    def power_off(self) -> None:
        response = self._serial_port.execute_command_with_response("F")

        time.sleep(self.POWER_OFF_DURATION_s)
        if "OFF" not in response:
            raise Exception(f"Power off failed. Response: {response}")  # pylint: disable=broad-exception-raised


class InterfaceNotFoundException(Exception):
    """Interface not found exception"""


@dataclass
class DeviceInfo:  # pylint: disable=missing-class-docstring
    device_type_name: str
    device_pid: int
    device_vid: int
