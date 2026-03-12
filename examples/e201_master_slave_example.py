import time

from e201_test_platform.e201_versions.e201b import E201B
from e201_test_platform.e201_versions.e201q import E201Q

ref_data = {
    'resolution': 1184000,
}
orbis_data = {
    'resolution': 16384,
    'singleturn_bits': 14,
    'multiturn_bits': 16,
    'status_bits': 2,
    'crc_bits': 6,
}
e201_slave = None
e201_master = None


def check_framerate(n: int = 1000):

    t0 = time.perf_counter()

    for _ in range(n):
        pos_m = e201_master.read_position()
        pos_s = e201_slave.read()

    dt = time.perf_counter() - t0

    print(f"loop rate: {n/dt:.1f} Hz")
    print(f"encoder reads: {2*n/dt:.1f} Hz")

try:
    # Open master
    e201_master = E201B("COM78", orbis_data)
    e201_master.open()

    # Open slave
    e201_slave = E201Q("COM77", ref_data)
    e201_slave.open()

    e201_master.power_on()
    e201_slave.power_on()
    print(e201_slave.enable_trigger_slave())
    print(e201_master.enable_trigger_master())
    time.sleep(1)
    # print(e201_master.generate_trigger_pulse())

    master_pos = e201_master.read_position()
    slave_pos = e201_slave.read()
    print(e201_master.parse_position(master_pos))
    print(e201_slave.parse_slave_position(slave_pos))
    # check_framerate(1000)
    time.sleep(1)
    print(e201_master.disable_trigger())
    print(e201_slave.disable_trigger())

except Exception as e:
    import traceback

    traceback.print_exc()


finally:
    if e201_master:
        e201_master.power_off()
        e201_master.close()

    if e201_slave:
        e201_slave.power_off()
        e201_slave.close()