import traceback
from e201_gui.e201_drivers.master import Master
from e201_gui.e201_drivers.parser import Parser

devices_settings = {
    "e201_dut": {
        "type": "E2019B",
        "comport": "COM15",
    },
    "e201_ref": {
        "type": "E2019Q",
        "comport": None,
    },
}

encoder_data: dict = {
    "dut_settings": {
        "voltage": 5000,
        "communication": "biss",
        "is_rotary": True,
        "resolution": 16384,
        "singleturn_bits": 14,
        "multiturn_bits": 16,
        "status_bits": 2,
        "crc_bits": 6,
        "dut_bytes": 8,
        "polarity": 0,
        "phase": 1,
        "frequency": 1000,
    },
    "ref_settings": {"number_of_periods": 11840, "interpolation_factor": 100},
}


master = Master(devices_settings)
parser = Parser(encoder_data=encoder_data)

try:
    master.initialize_device(encoder_data["dut_settings"])

    # Enable synced sampling
    master.enable_synced_sampling()

    # Read sample
    sample = master.read_position()
    pos = parser.parse_position(sample)
    print(f"Converted position: {pos.get('Position') * parser.dut_resolution} [deg]")
    for key, value in pos.items():
        print(f"- {key}: {value}")

    # Check framerate
    framerate = master.check_framerate(100)
    print(f"Frame rate: {framerate} Hz")

    from e201_gui.e201_drivers.registers_presets import orbis_preset

    master.set_register_access(orbis_preset)

    print(master.read_registers(3, 0x04, 4, False))

    master.set_multiturn(0)
    master.set_multiturn(0)


except Exception:
    traceback.print_exc()

finally:
    master.disable_synced_sampling()
    master.close_connection()
