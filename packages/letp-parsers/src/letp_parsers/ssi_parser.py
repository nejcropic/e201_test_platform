from letp_parsers.base_dut_parser import BaseDUTParser


class SSIParser(BaseDUTParser):
    def __init__(self, settings):
        super().__init__(settings)

    def parse_dut_frame(self, data: bytes, dut_bytes: int) -> dict[str, int | bool]:

        total_bits = dut_bytes * 8

        # Evalyn always reads one extra leading bit
        position_bits = self.singleturn_bits + self.multiturn_bits
        if self.dut_settings.get("master") == "evalyn":
            position_bits += 1

        dut_frame = int.from_bytes(data[:dut_bytes], self.byteorder)

        # From your old working parser:
        # detailed_status is below status, both below position, all referenced from MSB side
        detailed_status_shift = total_bits - position_bits - self.status_bits - self.detailed_status_bits
        status_shift = total_bits - position_bits - self.status_bits

        detailed_status = (dut_frame >> detailed_status_shift) & ((1 << self.detailed_status_bits) - 1)
        status = (dut_frame >> status_shift) & ((1 << self.status_bits) - 1)

        position_raw = dut_frame >> (self.status_bits + self.detailed_status_bits)

        # Remove Evalyn extra bit if present
        if self.dut_settings.get("master") == "evalyn":
            position_raw &= (1 << position_bits) - 1

        singleturn_position = position_raw & ((1 << self.singleturn_bits) - 1)

        multiturn_position = 0
        if self.multiturn_bits > 0:
            multiturn_position = (position_raw >> self.singleturn_bits) & ((1 << self.multiturn_bits) - 1)

        return {
            "Position": singleturn_position,
            "Multiturn": multiturn_position,
            "Status": status,
            "Detailed_status": detailed_status,
            "CRC": False,
        }
