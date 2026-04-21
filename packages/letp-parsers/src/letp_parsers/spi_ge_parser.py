from letp_parsers.spi_parser import SPIParser


class SPIGEParser(SPIParser):
    def __init__(self, settings):
        super().__init__(settings)

    def parse_dut_frame(self, data, dut_bytes):
        dut_frame = int.from_bytes(data[:5], "big")
        total_bits = 24

        response = self.singleturn_bits + self.status_bits
        position_raw = self.get_bits(dut_frame, total_bits - response, response)

        # Singleturn, status
        status = position_raw & ((1 << self.status_bits) - 1)
        singleturn_position = self.get_bits(position_raw, self.status_bits, self.singleturn_bits)

        return {
            "Position": singleturn_position,
            "Multiturn": 0,
            "Status": status,
            "CRC": False,
        }
