
import serial
import serial.tools.list_ports  # type: ignore #import serial module


def find_stm_electronics() :
    """Find port corresponding to the provided VID"""
    stm_ports = []
    for port in serial.tools.list_ports.comports():
        if port.manufacturer == "STMicroelectronics.":
            stm_ports.append(port.name)

    return stm_ports

