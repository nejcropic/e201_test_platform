from e201_test_platform.e201 import E201

class E201S(E201):
    def __init__(self, comport: str, encoder_data: dict):
        super().__init__(comport, encoder_data)

        self.communication = encoder_data.get("communication")

        if self.communication == "biss":
            self._read_cmd = "4"
        else:
            self._read_cmd = ">"

    def read_position(self):
        return self._serial_port.execute_command_with_response(self._read_cmd)