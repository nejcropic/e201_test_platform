from letp_parsers.base_dut_parser import BaseDUTParser


class HDSLParser(BaseDUTParser):
    def __init__(self, settings):
        super().__init__(settings)

    def parse_dut_frame(self, data, dut_bytes):
        """
        'data' samples in tl_et are in following order:
        [dut(fastpos), dut(cnt), dut(online_statusd), ref({'position': [_,_,..], 'dut_bytes': xx}) ]
        """
        dut_res = self.singleturn_bits + self.multiturn_bits

        # Extract data
        pgt13_pos_data, _, pgt13_statusd = data[0], data[1], data[2]

        # Data to return
        singleturn_position = pgt13_pos_data & ((2**self.singleturn_bits) - 1)
        multiturn_position = (pgt13_pos_data >> (dut_res - self.multiturn_bits)) & ((2**self.multiturn_bits) - 1)
        status = pgt13_statusd  # TODO: This is only online status... needs to be fixed!
        reference_position = int.from_bytes(list(data[3])[-16:-8], "little", signed=False)
        timer = int.from_bytes(list(data[3])[-8:], "big", signed=False)

        return {
            "Position": singleturn_position,
            "Multiturn": multiturn_position,
            "Status": status,
            "Reference": reference_position,
            "Timer": timer,
            "CRC": False,
        }
