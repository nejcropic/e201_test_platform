from letp_parsers.base_dut_parser import BaseDUTParser


class ABZParser(BaseDUTParser):
    def __init__(self, settings):
        super().__init__(settings)

    def parse_dut_frame(self, data, dut_bytes):
        dut_frame = int.from_bytes(data[:dut_bytes], self.byteorder)

        return {
            "Position": dut_frame,
            "CRC": False,
        }
