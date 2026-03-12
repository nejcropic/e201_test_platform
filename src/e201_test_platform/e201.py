import time
from e201_test_platform.serial_port import SerialPort, DeviceNotFound
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Union, cast
import serial
import serial.tools.list_ports  # type: ignore #import serial module


class E201(ABC):
    def __init__(self, comport, encoder_data):
        try:
            self._serial_port = SerialPort(comport)
        except DeviceNotFound:
            available_comports = []
            for port in serial.tools.list_ports.comports():
                if "STMicroelectronics" in port.manufacturer:
                    available_comports.append(port.device)
            raise ConnectionError(f"Wrong comport! Detected E201 comports: {available_comports}")

        self.resolution = encoder_data.get("resolution")

    def initialize(self):
        self.open()
        self.power_on()
        self.disable_trigger()

    def read(self) -> str:
        return self._serial_port.read()

    def write(self, command: Union[str, bytearray]) -> None:
        return self._serial_port.write(command)

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

    def power_on(self):
        self._serial_port.execute_command_with_response("N")
        time.sleep(1)
        self.initialized = True

    def power_off(self):
        self._serial_port.execute_command_with_response("F")
        time.sleep(1)
        self.initialized = False

    @abstractmethod
    def read_position(self):
        raise NotImplementedError

    def enable_trigger_master(self):
        return self._serial_port.execute_command_with_response("TM")

    def enable_trigger_slave(self):
        return self._serial_port.execute_command_with_response("TS")

    def set_trigger_data_format(self, data_format: int):
        return self._serial_port.execute_command_with_response(f"T{data_format}")

    def disable_trigger(self):
        return self._serial_port.execute_command_with_response("Tx")

    def generate_trigger_pulse(self):
        return self._serial_port.execute_command_with_response("TT")

    def check_framerate(self, n: int = 1000):
        start_time = time.perf_counter()
        for i in range(n):
            self.read_position()
        evaluation = time.perf_counter() - start_time
        print(f"Frame rate: {n / evaluation} Hz")

    def get_angle(self, position_counts: int):
        angle_deg = (position_counts % self.resolution) * (360.0 / self.resolution)
        return angle_deg