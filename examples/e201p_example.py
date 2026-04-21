from e201_gui.e201_versions.e201p import E201P
from e201_gui.parser import SPIParser

voltage = 5000

e201 = E201P("COM79", voltage)
e201.open()

encoder_data = {
    "dut_settings": {
        "resolution": 2097152,
        "singleturn_bits": 21,
        "multiturn_bits": 16,
        "status_bits": 2,
        "crc_bits": 8,
        "dut_bytes": 6,
    },
    "ref_settings": {"number_of_periods": 11840, "interpolation_factor": 100},
}

spi_parser = SPIParser(encoder_data)

try:
    e201.initialize()

    print(e201.get_version())
    print(e201.set_communication_protocol("SPI_EncoLink"))
    print(e201.initialize_encolink_library())

    print("Reading position... ")
    pos = e201.read_position()
    raw = e201.parse_raw_position(pos)

    parsed = spi_parser.parse_position(pos)
    print(parsed)
    # print(pos)
    # print("- Position [deg]:", pos.get('angle'))
    # print("- Status:", pos.get('status'))
    # print("- Position [counts]:", pos.get('singleturn'))
    # e201.check_framerate()

except Exception:
    import traceback

    traceback.print_exc()

finally:
    e201.power_off()
    e201.close()
