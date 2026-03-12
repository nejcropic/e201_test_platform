import time
from e201_test_platform.e201 import E201
from e201_test_platform.register_map import REGISTER_MAP


class E201B(E201):
    def __init__(self, comport: str, encoder_data: dict):
        super().__init__(comport, encoder_data)
        self.singleturn = encoder_data.get("singleturn_bits")
        self.multiturn_bits = encoder_data.get("multiturn_bits")
        self.status_bits = encoder_data.get("status_bits")
        self.crc_bits = encoder_data.get("crc_bits")

    def read_position(self) -> str:
        return self._serial_port.execute_command_with_response("4")

    def parse_position(self, position: str,) -> dict:
        """
        Parse 16-hex-character BiSS frame returned by E201-9B command '4'.

        Expected frame layout (MSB first):
        [ Position | Status | CRC ]
        """
        hex_position = position.strip()

        if len(hex_position) != 16:
            raise ValueError(f"Expected 16 hex chars, got {len(hex_position)}")


        # Convert to 64-bit integer
        dut_frame = int(hex_position, 16)
        frame_length = 64

        singleturn_bits = self.singleturn
        multiturn_bits = self.multiturn_bits
        status_bits = self.status_bits
        crc_bits = self.crc_bits
        dut_res = singleturn_bits + multiturn_bits

        # CRC
        crc_data_bits = dut_res + status_bits
        received_crc = (dut_frame >> (64 - dut_res - status_bits - crc_bits)) & ((1 << crc_bits) - 1)
        received_crc ^= ((1 << crc_bits) - 1)
        crc_frame = (dut_frame >> (frame_length - dut_res - status_bits)) & ((1 << crc_data_bits) - 1)
        # crc_error = self.crc_check(crc_frame, crc_data_bits, 0x43, received_crc)

        # Status
        status = (dut_frame >> (frame_length - dut_res - status_bits)) & ((1 << status_bits) - 1)

        # Position
        position = dut_frame >> (frame_length - dut_res)
        singleturn_position = position  & ((1 << singleturn_bits) - 1)

        multiturn_position = (position >> singleturn_bits) & ((1 << multiturn_bits) - 1)

        angle = self.get_angle(singleturn_position)
        return {
            "raw_hex": hex_position,
            "multiturn": multiturn_position,
            "singleturn": singleturn_position,
            "angle": angle,
            "status": status,
            "crc": False,
        }

    def write_register(self, value: int, address: int | str) -> str:
        addr = self._to_addr(address)

        if not (0 <= value <= 255):
            raise ValueError("Register value must be 0–255")
        if not (0 <= addr <= 127):
            raise ValueError("Register address must be 0–127")

        cmd = f"Ws{value:03d}:{addr:03d}"
        return self._serial_port.execute_command_with_response(cmd)

    def write_registers(self, value: int, bank: int, address: int | str, num_regs: int, signed: bool = False):
        """
        Write multi-byte register value.
        """
        addr = self._to_addr(address)

        value_bytes = int(value).to_bytes(
            num_regs,
            byteorder="big",
            signed=signed
        )
        self.select_bank(bank)
        for i, b in enumerate(value_bytes):
            self.raise_on_status(self.write_register(b, addr + i))

    def read_registers_params(self, reg: str) -> dict:
        r = REGISTER_MAP[reg]
        return self.read_registers(r["bank"], r["addr"], r["length"], r["signed"])

    def write_registers_params(self, value: int, reg: str):
        r = REGISTER_MAP[reg]
        self.write_registers(value,r["bank"], r["addr"],r["length"],  r["signed"])

    def read_registers(self, bank: int, address: int | str, num_regs: int, signed: bool = False) -> dict:
        self.select_bank(bank)
        resp = self._read_register(address, num_regs)
        parsed = self.parse_read_response(resp)
        self.raise_on_status(parsed["status"])

        return self.decode_register_value(parsed["data_hex"], num_regs, signed)

    def _read_register(self, address: int | str, length: int ) -> str:
        addr = self._to_addr(address)

        if not (1 <= length <= 64):
            raise ValueError("Length must be 1–64")
        if not (0 <= addr <= 127):
            raise ValueError("Register address must be 0–127")

        cmd = f"R{length:02d}:{addr:03d}"
        return self._serial_port.execute_command_with_response(cmd)

    @staticmethod
    def parse_read_response(resp: str) -> dict:
        response = resp.strip().split(":")
        return {"status": response[0],
                "details": response[1],
                "data_hex": response[2]}

    @staticmethod
    def raise_on_status(status: str) -> None:
        if status == "0":
            return
        if status == "1":
            raise RuntimeError("End of bank reached")
        if status == "2":
            raise RuntimeError("CRC error or incorrect data length")
        if status == "3":
            raise RuntimeError("Address > 127 or number of bytes > 64 or zero")
        if status == "4":
            raise RuntimeError("Timeout")
        raise RuntimeError(f"Unknown status: {status}")

    def read_detailed_status(self) -> str:
        return self.read_registers_params('detailed_status')['str_response']

    def set_multiturn(self, value: int) -> int:
        self.write_registers_params(value, "multiturn_set")
        self._write_key()
        self.raise_on_status(self.write_register(0x6D, REGISTER_MAP['multiturn_apply']['addr']))

        verify = self.read_registers_params("multiturn_set")
        return verify["int_response"]

    def set_position_offset(self, value: int) -> int:
        # Write offset
        self.write_registers_params(value, "position_offset")

        # Read offset
        verify = self.read_registers_params("position_offset")

        return verify["int_response"]

    def select_bank(self, bank: int):
        self.raise_on_status(self.write_register(bank, 0x40))

    def safe_to_flash(self):
        self.write_registers_params(0, 'save_to_flash')

    def _write_key(self):
        self.raise_on_status(self.write_register(0xCD, 0x48))

    @staticmethod
    def decode_register_value(data_hex: str, length: int,  signed: bool) -> dict:
        raw_bytes = bytes.fromhex(data_hex)
        data_int = int.from_bytes(raw_bytes, byteorder="big", signed=signed)
        data_str = format(int.from_bytes(raw_bytes, "big", signed=False), f"0{length * 8}b")
        return {
            'hex_response': data_hex,
            'int_response': data_int,
            'str_response': data_str
        }

    @staticmethod
    def _to_addr(address: int | str) -> int:
        # Accept 74, "74", "0x4A", "4A"
        if isinstance(address, int):
            return address
        s = address.strip().lower()
        if s.startswith("0x"):
            return int(s, 16)
        # if it's pure digits -> decimal, otherwise hex
        return int(s, 10) if s.isdigit() else int(s, 16)
