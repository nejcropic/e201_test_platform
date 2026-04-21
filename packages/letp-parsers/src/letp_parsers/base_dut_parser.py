from abc import abstractmethod, ABC
from typing import Literal

ByteOrder = Literal["little", "big"]


class BaseDUTParser(ABC):
    byteorder: ByteOrder = "big"

    def __init__(self, dut_settings):
        self.dut_settings = dut_settings
        self.dut_length_bits = None

        if self.dut_settings.get("error_warning_bits", None) is not None:
            self.dut_settings["status_bits"] = self.dut_settings["error_warning_bits"]

        self.singleturn_bits = self.dut_settings.get("singleturn_bits", 0)
        self.multiturn_bits = self.dut_settings.get("multiturn_bits", 0)
        self.status_bits = self.dut_settings.get("status_bits", 0)
        self.detailed_status_bits = self.dut_settings.get("detailed_status_bits", 0)
        self.crc_bits = self.dut_settings.get("crc_bits", 0)
        self.persistent_bits = self.dut_settings.get("persistent_bits", 0)
        self.incremental_position_bits = self.dut_settings.get("incremental_position_bits", 0)

    @abstractmethod
    def parse_dut_frame(self, dut_frame: bytes, dut_bytes: int) -> dict:
        pass

    @staticmethod
    def get_bits(frame, offset, width):
        """
        offset = number of bits to shift right
        width = number of bits to extract
        """
        return (frame >> offset) & ((1 << width) - 1)

    @staticmethod
    def crc_check(data: int, data_bits: int, polynomial: int, received_crc: int) -> bool:
        """
        CRC check, MSB-first, width inferred from polynomial.
        Returns True if CRC ERROR, False if OK.
        """
        crc_width = polynomial.bit_length() - 1

        # Append CRC to data
        reg = (data << crc_width) | received_crc

        total_bits = data_bits + crc_width
        poly = polynomial << (total_bits - crc_width - 1)

        for i in range(data_bits):
            if reg & (1 << (total_bits - 1 - i)):
                reg ^= poly >> i

        remainder = reg & ((1 << crc_width) - 1)

        return remainder != 0
