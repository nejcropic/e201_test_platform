from letp_parsers.parsers_imports import get_protocol_parser


class Parser:
    def __init__(self, encoder_data):
        self.dut_settings = encoder_data.get("dut_settings")
        self.ref_settings = encoder_data.get("ref_settings")
        self.dut_resolution = self.get_dut_resolution()
        self.ref_resolution = self.get_ref_resolution()
        self.dut_parser = get_protocol_parser(self.dut_settings.get("communication"))(self.dut_settings)

    def update_parser(self, encoder_data):
        self.dut_settings = encoder_data.get("dut_settings")
        self.ref_settings = encoder_data.get("ref_settings")
        self.dut_resolution = self.get_dut_resolution()
        self.ref_resolution = self.get_ref_resolution()

    def parse_position(self, data: bytes) -> dict[str, int]:
        if isinstance(data, list):
            data = bytes(data)

        dut_bytes = self.dut_settings["dut_bytes"]
        ref_frame = int.from_bytes(data[dut_bytes : dut_bytes + 4], "big")
        timestamp_frame = int.from_bytes(data[dut_bytes + 4 : dut_bytes + 12], "little")

        ref_frame = int(ref_frame % self.get_ref_counts())
        parsed_frame = self.dut_parser.parse_dut_frame(data, dut_bytes)
        parsed_frame["Reference"] = ref_frame
        parsed_frame["Timer"] = timestamp_frame

        return parsed_frame

    def get_ref_counts(self):
        if not self.dut_settings.get("is_rotary"):
            return self.ref_settings.get("period") * self.ref_settings.get("interpolation_factor")
        else:
            return self.ref_settings["number_of_periods"] * self.ref_settings["interpolation_factor"]

    def get_dut_resolution(self):
        return 360 / self.dut_settings.get("resolution")

    def get_ref_resolution(self):
        return 360 / (self.ref_settings.get("number_of_periods") * self.ref_settings.get("interpolation_factor"))

    @staticmethod
    def get_bits(frame, offset, width):
        """
        offset = number of bits to shift right
        width = number of bits to extract
        """
        return (frame >> offset) & ((1 << width) - 1)

    @staticmethod
    def crc_check(data: int, data_bits: int, polynomial: int, received_crc: int) -> bool:
        """
        CRC check, MSB-first, width inferred from polynomial.
        Returns True if CRC ERROR, False if OK.
        """
        crc_width = polynomial.bit_length() - 1

        # Append CRC to data
        reg = (data << crc_width) | received_crc

        total_bits = data_bits + crc_width
        poly = polynomial << (total_bits - crc_width - 1)

        for i in range(data_bits):
            if reg & (1 << (total_bits - 1 - i)):
                reg ^= poly >> i

        remainder = reg & ((1 << crc_width) - 1)

        return remainder != 0
