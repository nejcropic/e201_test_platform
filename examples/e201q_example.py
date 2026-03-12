import time

from e201_test_platform.e201_versions.e201q import E201Q

ref_data = {
    'resolution': 1184000,
}
e201 = E201Q("COM77", ref_data)
e201.open()

try:
    e201.power_on()

    print(e201.get_version())
    print("Pos:", e201.read_position())

    pos = e201.parse_position(e201.read_position())
    pos_deg = e201.get_angle(pos.get('singleturn'))
    print("Pos deg:",pos_deg)

    e201.check_framerate()

except Exception:
    import traceback

    traceback.print_exc()

finally:
    e201.power_off()
    e201.close()
