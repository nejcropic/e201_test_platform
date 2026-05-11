from e201_gui.motor_drivers.epos_api.EPOS_drv import EPOSMotor
from e201_gui.motor_drivers.motor_base import MotorBase


class EPOS(MotorBase):
    gear_ratio = 3
    steps_per_rotation = 4000

    def __init__(self):
        self.motor = EPOSMotor(acceleration=1000)
        self.enable_profile_velocity_mode()

    def enable(self):
        self.motor.set_enable_state()

    def disable(self):
        self.motor.set_disable_state()

    def set_speed(self, speed):
        self.motor.move_with_velocity(int(speed * self.gear_ratio))

    def stop(self):
        self.motor.soft_stop()

    def get_velocity(self):
        return self.motor.get_velocity() / self.gear_ratio

    def enable_profile_velocity_mode(self):
        self.motor.activate_profile_velocity_mode()

    def disconnect(self):
        self.motor.close_device()
