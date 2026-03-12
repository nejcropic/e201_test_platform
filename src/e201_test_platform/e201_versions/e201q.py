from e201_test_platform.e201 import E201

class E201Q(E201):
    def __init__(self, comport: str, encoder_data: dict):
        super().__init__(comport, encoder_data)

    def read_position(self) -> str:
        return self._serial_port.execute_command_with_response(">")

    def parse_slave_position(self, position: str) -> dict:
        hex_position = position.strip().split("=")[1]

        if len(hex_position) != 8:
            raise ValueError(f"Expected 8 hex chars, got {len(hex_position)}: {hex_position!r}")

        # Split into 2 words
        n_hex = hex_position[0:8]

        def u32_from_hex(h: str) -> int:
            return int(h, 16) & 0xFFFFFFFF

        def s32_from_hex(h: str) -> int:
            u = u32_from_hex(h)
            return u - 0x100000000 if (u & 0x80000000) else u

        n = s32_from_hex(n_hex)

        angle = self.get_angle(n)
        return {
            "raw_hex": hex_position,
            "singleturn": n,
            "angle": angle,
        }


    def parse_position(self, position: str) -> dict:
        """
        Parse 24-hex-character frame returned by E201-9Q command '>'.

        Format (3x 8-hex groups, big-endian):
          nnnnnnnn rrrrrrrr ssssssss

        where:
          n = encoder count (signed 32-bit)
          r = count when reference/index last seen (signed 32-bit)
          s = status (unsigned 32-bit; typically 0/1)
        """
        hex_position = position.strip()

        if len(hex_position) != 24:
            raise ValueError(f"Expected 24 hex chars, got {len(hex_position)}: {hex_position!r}")

        # Split into 3 words
        n_hex = hex_position[0:8]
        r_hex = hex_position[8:16]
        s_hex = hex_position[16:24]

        def u32_from_hex(h: str) -> int:
            return int(h, 16) & 0xFFFFFFFF

        def s32_from_hex(h: str) -> int:
            u = u32_from_hex(h)
            return u - 0x100000000 if (u & 0x80000000) else u

        n = s32_from_hex(n_hex)
        r = s32_from_hex(r_hex)
        s = u32_from_hex(s_hex)

        angle = self.get_angle(n)
        return {
            "raw_hex": hex_position,
            "singleturn": n,
            "ref_last": r,
            "status": s,
            "angle": angle,
        }
