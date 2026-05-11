"""
COPYRIGHT(c) 2020 RLS d.o.o, Pod vrbami 2, 1218 Komenda, Slovenia

file:      linvol_parser.py
brief:     Class for parsing analog signals acquired with ADC on Evalyn.
author(s): Nejc Ropič
date:      24.4.2026

details:   Provides parsing of eight adc probes from Evalyn.

"""

from letp_parsers.base_dut_parser import BaseDUTParser


class LinVolParser(BaseDUTParser):
    def __init__(self, settings):
        super().__init__(settings)

    def parse_dut_frame(self, dut_frame: bytes | list, dut_bytes):
        analog = dut_frame[self.dut_settings["analog_adc"] - 1]

        return {
            "Position": analog,
            "Analog": analog,
            "CRC": False,
        }
