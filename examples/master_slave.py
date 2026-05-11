import traceback
from e201_gui.e201_drivers.master import Master
from e201_gui.e201_drivers.parser import Parser
from e201_gui.helpers.file_handlers import load_yaml
from e201_gui.e201_drivers.registers_preset import get_registers_preset

# Set parameters
preset = get_registers_preset("ASKO Encolink")  # register preset
devices_settings = {  # Set device settings
    "e201_dut": {
        "type": "E2019P",
        "comport": "COM80",
    },
    "e201_ref": {
        "type": "E2019Q",
        "comport": "COM76",
    },
}
encoder_data: dict = load_yaml("last_settings.yaml")  # Load encoder data from .yaml file


# Set Master and Parser
master = Master(devices_settings)
parser = Parser(encoder_data=encoder_data)

try:
    # Initialize DUT device
    master.initialize_device(encoder_data["dut_settings"])

    # Enable synced sampling
    master.enable_synced_sampling()

    # Read sample
    sample = master.read_position()
    pos = parser.parse_position(sample)
    print("\nEncoder position read:")
    print(f"- Converted position: {pos.get('Position') * parser.dut_resolution} [deg]")
    for key, value in pos.items():
        print(f"- {key}: {value}")

    # Check framerate
    framerate = master.check_framerate(100)
    print(f"\nFrame rate: {framerate} Hz")

    # Read registers - register access only in 'P' and 'B' version
    if master.dut.__class__.__name__ in ["E2019B", "E2019P"]:
        master.set_register_access(preset)
        response = master.read_registers_params("detailed_status")
        print("\nDetailed status read:")
        for key, value in response.items():
            print(f"- {key.capitalize().replace('_', ' ')}: {value}")

except Exception:
    traceback.print_exc()

finally:
    master.disable_synced_sampling()
    master.close_connection()
