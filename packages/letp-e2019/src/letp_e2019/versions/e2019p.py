"""
COPYRIGHT(c) 2020 RLS d.o.o, Pod vrbami 2, 1218 Komenda, Slovenia

file:      e2019p.py
brief:     E2019P interface for P911 communication with protocol and EncoLink support.
author(s): Nejc Ropič
date:      17.4.2026

details:   Provides position readout, protocol configuration (SPI, EncoLink, PWM),
           and register access via EncoLink mode with automatic initialization
           and restoration of device state.
"""

from letp_e2019.e2019 import E2019Power
from letp_e2019.e2019_helpers import e2019p_available_freq


class E2019P(E2019Power):
    type: str = "P"
    communication_protocols = {"SPI_EncoLink": "Ce", "SPI": "Cp", "PWM": "Cw"}

    def __init__(self, com):
        """Initialize E2019P device with protocol settings and defaults."""
        super().__init__(com)
        self.available_freq = e2019p_available_freq
        self._trigger_last_state = False
        self.bytes = 6
        self.read_command = "?06:000"

    def set_communication_protocol(self, protocol: str) -> str:
        """
        Set communication protocol.

        Args:
            protocol: One of supported protocol keys.

        Returns:
            Device response.
        """
        if protocol not in self.communication_protocols.keys():
            raise ValueError(f"Supported protocols are: {self.communication_protocols.keys()}")

        return self.execute_command_with_response(self.communication_protocols[protocol])

    def set_clock_settings(self, polarity: int, phase: int):
        """
        Configure SPI clock polarity and phase.

        Args:
            polarity: Clock polarity (0/1).
            phase: Clock phase (0/1).
        """
        self.execute_command_with_response(f"G{int(polarity)}:{int(phase)}")

    def initialize_encolink_library(self):
        """
        Initialize EncoLink library.

        Returns:
            dict: Contains version, frame size, and part number.
        """
        response = self.execute_command_with_response("j")
        return {"version": response[0], "bytes_in_frame": response[1], "part_number": response[-16:]}

    def _initialize_encolink(self):
        """Prepare device for EncoLink register access (disables trigger if needed)."""
        if self._trigger_enabled:
            self.disable_trigger()
            self._trigger_last_state = True

        response = self.set_communication_protocol("SPI_EncoLink")
        if response != "SPI_ENCOLINK_MODE":
            raise ValueError("SPI EncoLink not established!")

        self.initialize_encolink_library()

    def _deinitialize_encolink(self):
        """Restore SPI mode and previous trigger state after EncoLink usage."""
        self.set_communication_protocol("SPI")
        if self._trigger_last_state:
            self.enable_trigger_master()
            self._trigger_last_state = False

    def write_registers(self, value: int, bank: int, address: int | str, length: int, is_signed: bool = False):
        """
        Write registers using EncoLink protocol.

        Args:
            value: Value to write.
            bank: Unused (kept for API compatibility).
            address: Register address.
            length: Number of bytes.
            is_signed: Unused.
        """
        self._initialize_encolink()
        self._write_register(value, address, length)
        self._deinitialize_encolink()

    def read_registers(self, bank: int, address: int | str, length: int, is_signed: bool = False):
        """
        Read registers using EncoLink protocol.

        Args:
            bank: Unused (kept for API compatibility).
            address: Register address.
            length: Number of bytes.
            is_signed: Interpret data as signed (not applied here).

        Returns:
            Raw bytes from device.
        """
        self._initialize_encolink()
        data = self._read_register(address, length, is_signed)
        self._deinitialize_encolink()
        return data

    def _write_register(self, value: int, address: int, length: int):
        """
        Write register via EncoLink command.

        Args:
            value: Value to write.
            address: Register address.
            length: Number of bytes.
        """
        status = self.execute_command_with_response(
            f"W:{self._to_hex(length, 4)}:{self._to_hex(address, 8)}:{self._to_hex(value, 8)}"
        )
        self._raise_on_status(status)

    def _read_register(self, address: int | str, length: int, is_signed: bool):
        """
        Read register via EncoLink command.

        Args:
            address: Register address.
            length: Number of bytes.
            is_signed: Unused.

        Returns:
            Raw bytes read from device.
        """
        resp = self.execute_command_with_response(f"R:{self._to_hex(length, 4)}:{self._to_hex(address, 8)}")

        data = resp.split(":")[1]
        if data.startswith("0x"):
            data = data[2:]

        return bytes.fromhex(data)

    @staticmethod
    def _to_hex(address: int, num_characters: int) -> str:
        """
        Convert integer to zero-padded uppercase hex string.

        Args:
            address: Value to convert.
            num_characters: Output string length.

        Returns:
            Formatted hex string.
        """
        return f"{address:0{num_characters}X}"
