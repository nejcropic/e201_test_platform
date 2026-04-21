# PROTOCOL PARSERS
from letp_parsers.abz_parser import ABZParser
from letp_parsers.biss_parser import BISSParser
from letp_parsers.hdsl_parser import HDSLParser
from letp_parsers.uart_parser import UARTParser
from letp_parsers.spi_parser import SPIParser
from letp_parsers.ssi_parser import SSIParser
from letp_parsers.spi_ge_parser import SPIGEParser

supported_communications = ["abz", "biss", "linvol", "hdsl", "sincos", "ssi", "spi", "spi_ge", "uart"]


protocol_parsers = {
    "abz": ABZParser,
    "biss": BISSParser,
    "hdsl": HDSLParser,
    "ssi": SSIParser,
    "spi": SPIParser,
    "spi_ge": SPIGEParser,
    "uart": UARTParser,
}


DEPENDENT_COMMUNICATIONS = {
    "hdsl": ("hdsl", "spi"),
}


def get_protocol_parser(communication: str):
    if communication not in protocol_parsers:
        raise NotImplementedError(f"{communication} communication not supported!")

    return protocol_parsers[communication]
