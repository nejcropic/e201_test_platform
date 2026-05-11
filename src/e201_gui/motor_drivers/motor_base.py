from abc import ABC, abstractmethod


class MotorBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def set_speed(self, speed):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    @abstractmethod
    def disable(self):
        raise NotImplementedError

    @abstractmethod
    def enable(self):
        raise NotImplementedError

    @abstractmethod
    def disconnect(self):
        raise NotImplementedError

    @abstractmethod
    def get_velocity(self) -> float:
        raise NotImplementedError
