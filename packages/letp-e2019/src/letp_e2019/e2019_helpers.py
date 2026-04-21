read_commands: dict = {
    "E2019B": {
        "command": "4",
    },
    "E2019P": {
        "command": "?06:000",
    },
    "E2019Q": {
        "command": ">",
    },
    "E2019S": None,
}


e2019p_errors: dict = {
    "0x9": "OK",
    "0x26": "Invalid register address",
    "0x56": "Value out of range",
    "0x96": "Access denied",
    "0xEE": "Incorrect number of bytes (register length mismatch)",
    "0xF6": "Write access is locked",
    "0xF9": "CRC invalid on write",
    "0xE6": "CRC invalid on read",
}

e2019b_errors: dict = {
    "0": "OK",
    "1": "End of bank reached",
    "2": "CRC error or incorrect data length",
    "3": "Address > 127 or number of bytes > 64 or zero",
    "4": "Timeout",
}
