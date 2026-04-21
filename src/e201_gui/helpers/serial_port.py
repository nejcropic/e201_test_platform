from typing import Union, Any
import serial
import serial.tools.list_ports  # type: ignore #import serial module


class SerialPort:
    """Light weight wrapper around the `pyserial.Serial` object."""

    def __init__(self, vid_or_com: Union[int, str], baud_rate: int = 250_000, timeout_s: float = 0.5) -> None:
        if isinstance(vid_or_com, int):
            # look for a device with the appropriate Vendor IDentifier (VID)
            port = _find_port_by_vid(vid_or_com)
            if not port:
                raise DeviceNotFound(f"Device with VID: {vid_or_com} not found!")
            port_name = port.name
        elif isinstance(vid_or_com, str):
            port_name = vid_or_com
            available_ports = [p.device for p in serial.tools.list_ports.comports()]
            if port_name not in available_ports:
                raise DeviceNotFound(f"COM port '{port_name}' not found among: {available_ports}")
        else:
            raise TypeError("vid_or_com must be either an int (VID) or str (COM port name)")

        # configure and open the port
        self.ser = serial.Serial(port_name, baud_rate, timeout=timeout_s)

    def open(self) -> None:
        """Open the serial port"""
        if not self.is_open():
            self.ser.open()

    def close(self) -> None:
        """Close the serial port"""
        if self.is_open():
            self.ser.close()

    def is_open(self) -> bool:
        return self.ser.isOpen()

    def write(self, command: Union[str, bytes, bytearray]) -> None:
        """Write command (string / bytearray) via a serial port"""
        if not self.is_open():
            raise ClosedPort("Communication port is closed!")

        if isinstance(command, str):
            command = command.encode()
        elif isinstance(command, bytearray):
            pass
        else:
            raise TypeError("Command must be either a bytearray or a string!")
        self.ser.reset_output_buffer()
        self.ser.write(command)

    def read(self) -> str:
        """Read the response"""
        if not self.is_open():
            raise ClosedPort("Communication port is closed!")

        response = self.ser.read_until(b"\r")
        return response.decode().strip()

    def read_until_terminator(self, terminator: bytes) -> str:
        """Read the response"""
        if not self.is_open():
            raise ClosedPort("Communication port is closed!")

        response = self.ser.read_until(terminator)
        return response

    def execute_command_with_response(self, command: Union[str, bytearray]) -> str:
        """Write command and read the response"""
        if not self.is_open():
            raise ClosedPort("Communication port is closed!")

        self.write(command)
        response = self.read()
        if "Error" in response or "error" in response:
            raise Exception(f"command: {command} failed with {response}")  # pylint: disable=broad-exception-raised
        return response


def _find_port_by_vid(vid: int) -> Any:
    """Find port corresponding to the provided VID"""
    for port in serial.tools.list_ports.comports():
        if port.vid == vid:
            return port
    return None


class ClosedPort(Exception):
    """The serial port has been closed"""


class DeviceNotFound(Exception):
    """Device has not be found"""
