from letp_e2019.e2019 import E2019


class E2019B(E2019):
    type: str = "B"
    available_freq = {
        10000: 0,
        5000: 1,
        3333: 2,
        2500: 3,
        2000: 4,
        1667: 5,
        1429: 6,
        1250: 7,
        1111: 8,
        1000: 9,
        909: 10,
        833: 11,
        769: 12,
        714: 13,
        667: 14,
        625: 15,
        500: 17,
        333: 18,
        250: 19,
        200: 20,
        167: 21,
        143: 22,
        125: 23,
        111: 24,
        100: 25,
        91: 26,
        83: 27,
        77: 28,
        71: 29,
        67: 30,
        63: 31,
    }

    def __init__(self, com):
        super().__init__(com)
        self.bank_selected = -1
        self.bytes = 8
        self.read_command = "4"

    def write_registers(self, value: int, bank: int, address: int | str, length: int, is_signed: bool = False):
        addr = self._to_addr(address)

        value_bytes = int(value).to_bytes(length, byteorder="big", signed=is_signed)
        self.select_bank(bank)
        for i, b in enumerate(value_bytes):
            self._raise_on_status(self.write_register(b, addr + i))

    def read_registers(self, bank: int, address: int | str, length: int, is_signed: bool = False) -> bytes:
        self.select_bank(bank)

        resp = self.read_register(address, length)
        parsed = self._parse_register_response(resp)

        self._raise_on_status(parsed["status"])

        response_bytes = bytes.fromhex(parsed.get("data_hex"))
        return response_bytes

    def write_register(self, value: int, address: int | str) -> str:
        addr = self._to_addr(address)

        if not (0 <= value <= 255):
            raise ValueError("Register value must be 0–255")
        if not (0 <= addr <= 127):
            raise ValueError("Register address must be 0–127")

        cmd = f"Ws{value:03d}:{addr:03d}"
        return self.execute_command_with_response(cmd)

    def read_register(self, address: int | str, length: int) -> str:
        addr = self._to_addr(address)

        if not (1 <= length <= 64):
            raise ValueError("Length must be 1–64")
        if not (0 <= addr <= 127):
            raise ValueError("Register address must be 0–127")

        cmd = f"R{length:02d}:{addr:03d}"
        return self.execute_command_with_response(cmd)

    def select_bank(self, bank: int):
        if bank != self.bank_selected:
            self._raise_on_status(self.write_register(bank, 0x40))
            self.bank_selected = bank

    @staticmethod
    def _to_addr(address: int | str) -> int:
        """
        Accepts '74', '74', '0x4A', '4A'
        Args:
            address (int | str): address

        Returns:
            address integer
        """
        if isinstance(address, int):
            return address

        s = address.strip().lower()
        if s.startswith("0x"):
            return int(s, 16)

        return int(s, 10) if s.isdigit() else int(s, 16)
