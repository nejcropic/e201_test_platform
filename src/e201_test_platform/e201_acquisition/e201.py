import serial
import serial.tools.list_ports
import time




import serial
import serial.tools.list_ports
import time


class E201:
    def __init__(self, port_name):
        self.ser = serial.Serial(
            port_name,
            baudrate=115200,
            timeout=0,
            write_timeout=0
        )
        self._buffer = bytearray()

    def send(self, cmd: str):
        self.ser.write(cmd.encode() + b"\r")

    def read_lines(self):
        data = self.ser.read(self.ser.in_waiting or 1)
        if data:
            self._buffer.extend(data)

        lines = []
        while b"\r\n" in self._buffer:
            line, _, self._buffer = self._buffer.partition(b"\r\n")
            lines.append(line.decode())
        return lines

    @staticmethod
    def _get_comport(version):
        for port in list(serial.tools.list_ports.comports()):
            print(port.name, port.pid, port.vid, port.serial_number)
            if port.name == version:
                return port.name

        raise ConnectionRefusedError("No comport found")


