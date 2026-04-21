from letp_parsers.base_dut_parser import BaseDUTParser


class BISSParser(BaseDUTParser):
    def __init__(self, settings):
        super().__init__(settings)

    def parse_dut_frame(self, data, dut_bytes):
        dut_frame = int.from_bytes(data[:dut_bytes], self.byteorder)

        frame_bits = dut_bytes * 8
        dut_res = self.singleturn_bits + self.multiturn_bits + self.persistent_bits

        # CRC
        crc_data_bits = dut_res + self.status_bits
        received_crc = (dut_frame >> (frame_bits - dut_res - self.status_bits - self.crc_bits)) & (
            (1 << self.crc_bits) - 1
        )
        received_crc ^= (1 << self.crc_bits) - 1
        crc_frame = (dut_frame >> (frame_bits - dut_res - self.status_bits)) & ((1 << crc_data_bits) - 1)
        crc_error = self.crc_check(crc_frame, crc_data_bits, 0x43, received_crc)

        # Status
        status = (dut_frame >> (frame_bits - dut_res - self.status_bits)) & ((1 << self.status_bits) - 1)

        # Position
        position = (dut_frame >> (frame_bits - dut_res)) & ((1 << dut_res) - 1)
        singleturn_position = position & ((1 << self.singleturn_bits) - 1)
        multiturn_position = position >> self.singleturn_bits & ((1 << self.multiturn_bits) - 1)

        return {
            "Position": singleturn_position,
            "Multiturn": multiturn_position,
            "Status": status,
            "CRC": crc_error,
        }
