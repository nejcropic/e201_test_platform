import traceback
from e201_gui.e201_drivers.master import Master
from e201_gui.e201_drivers.parser import Parser
from e201_gui.helpers.file_handlers import load_yaml
from e201_gui.e201_drivers.registers_preset import get_registers_preset
from e201_gui.motor_drivers.supported_motors import get_supported_motor

# Set parameters
motor_speed = 20  # [rpm]
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

# Connect with motor
motor = get_supported_motor("EPOS")

try:
    # Initialize DUT device
    master.initialize_device(encoder_data["dut_settings"])
    master.set_register_access(preset)

    # Reset DUT to factory reset
    master.factory_reset()

    # Enable synced sampling
    master.enable_synced_sampling()

    # Enable motor and set speed
    print(f"\nRunning motor with {motor_speed} RPM..")
    motor.enable()
    motor.set_speed(motor_speed)

    # Check self calibration status
    response = master.read_self_calibration_finished()
    status_bef_int = response["response_int"]

    # Start self calibration
    print("\nPerforming self calibration procedure..")
    selfcal_timeout = 60 / motor_speed * 1.3 + 0.5  # calculate one rotation time and increase by 30%
    master.start_self_calibration()

    # Wait until self calibration is finished
    response = master.wait_while_selfcal_is_active(selfcal_timeout)
    status_aft_int = response["response_int"]  # type: ignore

    # Check increment bits
    increment_bits_before = status_bef_int & ((1 << 2) - 1)
    increment_bits_after = status_aft_int & ((1 << 2) - 1)
    if increment_bits_after <= increment_bits_before:
        print(f"Calibration failed: increment bits before/after - {increment_bits_before}/{increment_bits_after}")

    # Check self calibration status register
    response = master.read_self_calibration_finished()
    print("\nSelf calibration status:")
    for key, value in response.items():
        print(f"- {key.capitalize().replace('_', ' ')}: {value}")

    # Read calculated eccentricity shift
    if master.register_access["self_cal_ring_ecc"].execute:
        response = master.read_registers_params("self_cal_ring_ecc")
        print("\nEccentricity shift:")
        for key, value in response.items():
            print(f"- {key.capitalize().replace('_', ' ')}: {value}")

except Exception:
    traceback.print_exc()

finally:
    motor.stop()
    motor.disable()
    master.disable_synced_sampling()
    master.close_connection()
