from e201_gui.e201_versions.e201b import E201B
from e201_gui.parser import BISSParser


e201 = E201B("COM78")
e201.open()

encoder_data = {
    "dut_settings": {
        "resolution": 2097152,
        "singleturn_bits": 21,
        "multiturn_bits": 16,
        "status_bits": 2,
        "crc_bits": 8,
    },
    "ref_settings": {"number_of_periods": 11840, "interpolation_factor": 100},
}


biss_parser = BISSParser(encoder_data)

try:
    e201.power_on()

    print(e201.get_version())

    # print("Pos:", e201.read_position())

    # print("MT set:", e201.set_multiturn(0))  # includes verify readback
    # print("MT set:", e201.set_multiturn(0))  # includes verify readback
    #
    # time.sleep(2)
    #
    # print("Position offset:", e201.set_position_offset(50))  # includes verify readback
    #
    # print("Detailed status:", e201.read_detailed_status())

    print("Reading position... ")
    pos = e201.parse_position(e201.read_position())

    print(pos)
    # print("- Position [deg]:", pos.get('angle'))
    # print("- Status:", pos.get('status'))
    # print("- Position [counts]:", pos.get('singleturn'))
    # e201.check_framerate()

except Exception:
    import traceback

    traceback.print_exc()

finally:
    # e201.power_off()
    e201.close()
