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

e2019p_available_freq = {
    94: 1,
    187: 2,
    375: 3,
    750: 4,
    1500: 5,
    3000: 6,
    6000: 7,
    12000: 8,
}


e2019b_errors: dict = {
    "0": "OK",
    "1": "End of bank reached",
    "2": "CRC error or incorrect data length",
    "3": "Address > 127 or number of bytes > 64 or zero",
    "4": "Timeout",
}

e2019b_available_freq: dict = {
    10000: 0,
    5000: 1,
    3333: 2,
    2500: 3,
    2000: 4,
    1667: 5,
    1429: 6,
    1250: 7,
    1111: 8,
    1000: 9,
    909: 10,
    833: 11,
    769: 12,
    714: 13,
    667: 14,
    625: 15,
    500: 17,
    333: 18,
    250: 19,
    200: 20,
    167: 21,
    143: 22,
    125: 23,
    111: 24,
    100: 25,
    91: 26,
    83: 27,
    77: 28,
    71: 29,
    67: 30,
    63: 31,
}

e2019s_available_freq: dict = {
    35: 1,
    70: 2,
    140: 3,
    280: 4,
    560: 5,
    1100: 6,
    2200: 7,
    4400: 8,
}
