from e201_test_platform.e201 import E201


class E201P(E201):
    def __init__(self, comport: str, encoder_data: dict):
        super().__init__(comport, encoder_data)
        self.encoder_data = encoder_data

    def read_position(self):
        return self._serial_port.execute_command_with_response("?04:000")

    def parse_position(self, data):
        singleturn_bits = self.dut_settings["singleturn_bits"]
        multiturn_bits = self.dut_settings.get("multiturn_bits", 0)
        error_warning_bits = self.dut_settings["error_warning_bits"]
        crc_bits = self.dut_settings["crc_bits"]
        offset_bits = 8

        dut_bytes = self.dut_length_bits // 8

        dut_frame = int.from_bytes(data[:dut_bytes], "big")
        ref_frame = int.from_bytes(data[-16:-8], "little")
        time_frame = int.from_bytes(data[-8:], "little")

        total_bits = dut_bytes * 8

        position_offset = total_bits - (singleturn_bits + multiturn_bits)
        position_raw = self.get_bits(
            dut_frame,
            position_offset,
            singleturn_bits + multiturn_bits,
        )

        # Singleturn, multiturn, status
        status = self.get_bits(dut_frame, crc_bits + offset_bits, error_warning_bits)
        singleturn_position = self.get_bits(position_raw, 0, singleturn_bits)
        multiturn_position = self.get_bits(position_raw, singleturn_bits, multiturn_bits) if multiturn_bits > 0 else 0

        return {
            "Position": singleturn_position,
            "Multiturn": multiturn_position,
            "Status": status,
            "Reference": ref_frame,
            "Timer": time_frame,
            "CRC": False,
        }