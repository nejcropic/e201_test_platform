from letp_e2019.e2019 import E2019Power


class E2019Q(E2019Power):
    def __init__(self, comport: str):
        super().__init__(comport)
        self.trigger_enabled = False
        self.bytes = 4
        self.read_command = ">"

    def read_position(self) -> str:
        if self.trigger_enabled:
            return self.read()
        return self._serial_port.execute_command_with_response(self.read_command)

    def set_read_command(self, communication: str):
        pass

    def read_clock_frequency(self):
        pass

    def set_clock_frequency(self, freq_khz: int):
        pass
