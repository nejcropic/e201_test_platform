from letp_parsers.base_dut_parser import BaseDUTParser


class SPIParser(BaseDUTParser):
    def __init__(self, settings):
        super().__init__(settings)
        self.dut_length_bits = self.multiturn_bits + 4 * 8

    def parse_dut_frame(self, data, dut_bytes):
        if self.dut_settings.get("evalyn", "evalyn") == "evalyn":
            dut_bytes = self.dut_length_bits // 8

        dut_frame = int.from_bytes(data[:dut_bytes], self.byteorder)
        total_bits = dut_bytes * 8

        # --- CRC (LSB side) ---
        received_crc = self.get_bits(dut_frame, 0, self.crc_bits)

        # --- STATUS (just above CRC) ---
        status = self.get_bits(dut_frame, self.crc_bits, self.status_bits)

        # --- POSITION (MSB side) ---
        position_bits = self.singleturn_bits + self.multiturn_bits
        position_offset = total_bits - position_bits
        position_raw = self.get_bits(dut_frame, position_offset, position_bits)

        singleturn_position = self.get_bits(position_raw, 0, self.singleturn_bits)

        multiturn_position = 0
        if self.multiturn_bits > 0:
            multiturn_position = self.get_bits(position_raw, self.singleturn_bits, self.multiturn_bits)

        # --- CRC CHECK ---
        data_bits = total_bits - self.crc_bits
        data_for_crc = dut_frame >> self.crc_bits  # remove CRC bits

        crc_error = self.crc_check(
            data=data_for_crc,
            data_bits=data_bits,
            polynomial=0x101,
            received_crc=received_crc,
        )

        return {
            "Position": singleturn_position,
            "Multiturn": multiturn_position,
            "Status": status,
            "CRC": not crc_error,  # True = OK
        }
