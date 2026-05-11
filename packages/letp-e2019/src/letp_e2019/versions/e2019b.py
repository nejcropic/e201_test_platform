"""
COPYRIGHT(c) 2020 RLS d.o.o, Pod vrbami 2, 1218 Komenda, Slovenia

file:      e2019b.py
brief:     E2019P interface for P911 communication with protocol.
author(s): Nejc Ropič
date:      17.4.2026

details:   Supports communication with commands supported for E2019B.
"""

from letp_e2019.e2019 import E2019
from letp_e2019.e2019_helpers import e2019b_available_freq


class E2019B(E2019):
    type: str = "B"

    def __init__(self, com):
        """Initialize E2019B device with frequency table, data size, and read command."""
        super().__init__(com)
        self.available_freq = e2019b_available_freq
        self.bytes = 8
        self.read_command = "4"

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
        addr = self._to_addr(address)

        value_bytes = int(value).to_bytes(length, byteorder="big", signed=is_signed)
        if address != 0x49:
            self.select_bank(bank)

        for i, b in enumerate(value_bytes):
            self._raise_on_status(self._write_register(b, addr + i))

    def read_registers(self, bank: int, address: int | str, length: int, is_signed: bool = False) -> bytes:
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
        if address != 0x49:
            self.select_bank(bank)

        resp = self._read_register(address, length)
        parsed = self._parse_register_response(resp)

        self._raise_on_status(parsed["status"])

        response_bytes = bytes.fromhex(parsed.get("data_hex"))
        return response_bytes

    def _write_register(self, value: int, address: int | str) -> str:
        """
        Write single register.

        Args:
            value: Byte value (0–255).
            address: Register address.

        Returns:
            Raw device response.
        """
        addr = self._to_addr(address)

        if not (0 <= value <= 255):
            raise ValueError("Register value must be 0–255")
        if not (0 <= addr <= 127):
            raise ValueError("Register address must be 0–127")

        cmd = f"Ws{value:03d}:{addr:03d}"
        return self.execute_command_with_response(cmd)

    def _read_register(self, address: int | str, length: int) -> str:
        """
        Read consecutive registers.

        Args:
            address: Start address.
            length: Number of bytes (1–64).

        Returns:
            Raw device response string.
        """
        addr = self._to_addr(address)

        if not (1 <= length <= 64):
            raise ValueError("Length must be 1–64")
        if not (0 <= addr <= 127):
            raise ValueError("Register address must be 0–127")

        cmd = f"R{length:02d}:{addr:03d}"
        return self.execute_command_with_response(cmd)

    def select_bank(self, bank: int):
        """Select register bank."""
        self._raise_on_status(self._write_register(bank, 0x40))

    @staticmethod
    def _to_addr(address: int | str) -> int:
        """
        Convert address from int/str (dec/hex) to integer.

        Args:
            address: Address in int, decimal string, or hex string.

        Returns:
            Integer address.
        """
        if isinstance(address, int):
            return address

        s = address.strip().lower()
        if s.startswith("0x"):
            return int(s, 16)

        return int(s, 10) if s.isdigit() else int(s, 16)
