"""
file:      e2019s.py
brief:     E2019S interface for P911 communication with BiSS and SSI support.
author(s): Nejc Ropič
date:      17.4.2026

details:   Supports position readout with selectable communication (BiSS/SSI),
           dynamic data length handling, and SSI word width configuration.
"""

from letp_e2019.e2019 import E2019Power
from letp_e2019.e2019_helpers import e2019s_available_freq


class E2019S(E2019Power):
    _available_communications = {"biss": "4", "ssi": ">"}

    def __init__(self, comm):
        super().__init__(comm)
        self.bytes = 4
        self.available_freq = e2019s_available_freq

    def set_read_command(self, communication: str):
        """
        Select communication mode and corresponding read command.

        Args:
            communication: 'biss' or 'ssi'.
        """
        if communication not in self._available_communications:
            raise ValueError(f"Communication not in available communications! {self._available_communications.keys()}")

        self.read_command = self._available_communications[communication]
        self.bytes = 8 if communication.lower() == "biss" else 4

    def set_word_width(self, word_width: int):
        """
        Set SSI word width (only valid for SSI mode).

        Args:
            word_width: Number of bits.
        """
        if self.read_command != ">":
            return

        command = f"B{word_width:02d}\r"
        self.execute_command_with_response(command)

        width_set = self.check_word_width().split(" ")[0]
        if word_width != int(width_set):
            raise ValueError(f"Word width not set! Word width: {width_set}")

    def check_word_width(self):
        """Read current SSI word width."""
        return self.execute_command_with_response("b")
