from e201_gui.motor_drivers.epos_api.epos import EPOS
from e201_gui.motor_drivers.motor_base import MotorBase
from e201_gui.motor_drivers.virtual_motor import VirtualMotor


supported_motor_types: dict = {"EPOS": EPOS, "Virtual": VirtualMotor}


def get_supported_motor(motor_type: str, gear_ratio: int) -> MotorBase:
    if motor_type not in supported_motor_types:
        raise NotImplementedError("Motor type not supported!")

    return supported_motor_types[motor_type](gear_ratio)
