import time

from e201_test_platform.e201_versions.e201b import E201B

orbis_data = {
    'resolution': 16384,
    'singleturn_bits': 14,
    'multiturn_bits': 16,
    'status_bits': 2,
    'crc_bits': 6,
}
e201 = E201B("COM78", orbis_data)
e201.open()

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
    print("- Position [deg]:", pos.get('angle'))
    print("- Status:", pos.get('status'))
    print("- Position [counts]:", pos.get('singleturn'))
    # e201.check_framerate()

except Exception:
    import traceback

    traceback.print_exc()

finally:
    # e201.power_off()
    e201.close()
