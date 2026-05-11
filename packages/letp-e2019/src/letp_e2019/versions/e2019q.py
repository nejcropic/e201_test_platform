"""
COPYRIGHT(c) 2020 RLS d.o.o, Pod vrbami 2, 1218 Komenda, Slovenia

file:      e2019q.py
brief:     E2019Q interface for P911 communication with protocol
author(s): Nejc Ropič
date:      17.4.2026

details:   Supports command-based (">") and trigger-based acquisition. In trigger mode,
           master generates pulses while slaves return position via passive read().
"""

from letp_e2019.e2019 import E2019Power


class E2019Q(E2019Power):
    def __init__(self, comport: str):
        super().__init__(comport)
        self.bytes = 4
        self.read_command = ">"

    def set_read_command(self, communication: str):
        raise ValueError("Set read command not supported in E2019Q!")

    def read_clock_frequency(self):
        raise ValueError("Read clock frequency not supported in E2019Q!")

    def set_clock_frequency(self, freq_khz: int):
        raise ValueError("Set clock frequency not supported in E2019Q!")
