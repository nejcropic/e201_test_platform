from letp_parsers.base_dut_parser import ByteOrder
from letp_parsers.base_dut_parser import BaseDUTParser


class UARTParser(BaseDUTParser):
    byteorder: ByteOrder = "little"

    def __init__(self, settings):
        super().__init__(settings)

    def parse_dut_frame(self, data, dut_bytes):
        dut_frame = int.from_bytes(data[16 : 16 + dut_bytes], self.byteorder)

        configuration_field = 8
        dut_res = self.singleturn_bits + self.incremental_position_bits

        singleturn_position = int.from_bytes(data[21:24], self.byteorder) >> (24 - self.singleturn_bits)
        incremental_position = int.from_bytes(data[18:21], self.byteorder) >> (24 - self.singleturn_bits)
        status = data[17]

        # CRC
        received_crc = (
            dut_frame >> (dut_bytes * 8 - dut_res - self.status_bits - configuration_field - self.crc_bits)
        ) & 0x3F
        crc_frame = (dut_frame >> (dut_bytes * 8 - dut_res - self.status_bits - configuration_field)) & 0xFFFFFFF
        crc_error = self.crc_check(crc_frame, dut_res + self.status_bits + configuration_field, 0x101, received_crc)

        return {
            "Position": singleturn_position,
            "Incremental": incremental_position,
            "Status": status,
            "CRC": crc_error,
        }
