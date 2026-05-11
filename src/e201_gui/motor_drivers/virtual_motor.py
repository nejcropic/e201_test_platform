from e201_gui.motor_drivers.motor_base import MotorBase


class VirtualMotor(MotorBase):
    def __init__(self):
        pass

    def set_speed(self, speed):
        print(f"[VIRTUAL MOTOR]: Motor speed set to {speed}")

    def stop(self):
        print("[VIRTUAL MOTOR]: Motor stop")

    def disable(self):
        print("[VIRTUAL MOTOR]: Motor disable")

    def enable(self):
        print("[VIRTUAL MOTOR]: Motor enable")

    def disconnect(self):
        print("[VIRTUAL MOTOR]: Motor disconnect")

    def get_velocity(self) -> float:
        return -999
