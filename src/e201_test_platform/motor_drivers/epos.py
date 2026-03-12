from e201_test_platform.motor_drivers.EPOS_drv import EPOSMotor

class EPOS:
    def __init__(self, config):
        self.gear_ratio = 3
        self.motor = EPOSMotor(acceleration=1000)
        self.enable_profile_velocity_mode()
        self.steps_per_rotation = 4000

    def enable(self):
        self.motor.set_enable_state()

    def disable(self):
        self.motor.set_disable_state()

    def set_speed(self, speed):
        self.motor.move_with_velocity(int(speed * self.gear_ratio))

    def step_forward(self, step_in_mm):
        self.motor.move_to_position(step_in_mm, False)

    def stop(self):
        self.soft_stop()

    def soft_stop(self):
        self.motor.soft_stop()

    def get_velocity(self):
        return self.motor.get_velocity() / self.gear_ratio

    def enable_profile_position_mode(self):
        self.motor.activate_profile_position_mode()

    def enable_profile_velocity_mode(self):
        self.motor.activate_profile_velocity_mode()

    def wait_while_active(self, timeout=10000):
        self.motor.wait_for_target_reached(timeout)

    def wait_for_speed_reached(self, timeout: float = 3000, err_thr: float = 5, throw: bool = False):
        """
        :param err_thr: Error threshold in rpm, which is allowed velocity error to break from this function.
        :param timeout: Maximum time to wait in this function in milliseconds. After this time is exceeded,
                        function will return, or raise exception, if 'throw' is set to True.
        :param throw: Specifies, whether RuntimeError should be raised if timeout is exceeded.

        :raise RuntimeError: If motor does not stop in specified time.
        """
        self.motor.wait_for_speed_reached(timeout, err_thr, throw)

    def go_home(self):
        pass

    def get_motor_position(self):
        return self.motor.get_motor_position()

    def disconnect(self):
        self.motor.close_device()

    def finish(self):
        self.disable()


if __name__ == "__main__":
    config = {"type": "Rotacijska_EPOS_naprava", "gear_ratio": 3}

    motor = EPOS(config)

    motor.enable()

    # VELOCITY MODE
    # motor.enable_profile_velocity_mode()
    # motor.set_speed(10)
    # time.sleep(2)

    # POSITION MODE
    motor.enable_profile_position_mode()
    last_pos_percentage = 0
    motor.stop()
    for i in [0.2, 0.4, 0.8, 1]:
        move = i - last_pos_percentage
        step = int(move * motor.steps_per_rotation * motor.gear_ratio)
        print(step)
        motor.step_forward(step)
        motor.wait_while_active(10000)
        last_pos_percentage = i

    motor.stop()
    motor.disable()
