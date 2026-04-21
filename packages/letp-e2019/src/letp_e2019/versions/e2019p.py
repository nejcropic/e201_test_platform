from letp_e2019.e2019 import E2019Power


class E2019P(E2019Power):
    type: str = "P"
    communication_protocols = {"SPI_EncoLink": "Ce", "SPI": "Cp", "PWM": "Cw"}
    available_freq = {
        94: 1,
        187: 2,
        375: 3,
        750: 4,
        1500: 5,
        3000: 6,
        6000: 7,
        12000: 8,
    }

    def __init__(self, com):
        super().__init__(com)
        self._trigger_last_state = False
        self.bytes = 6
        self.read_command = "?06:000"

    def set_communication_protocol(self, protocol: str) -> str:
        if protocol not in self.communication_protocols.keys():
            raise ValueError(f"Supported protocols are: {self.communication_protocols.keys()}")

        return self.execute_command_with_response(self.communication_protocols[protocol])

    def set_clock_settings(self, polarity: int, phase: int):
        self.execute_command_with_response(f"G{int(polarity)}:{int(phase)}")

    def initialize_encolink_library(self):
        response = self.execute_command_with_response("j")
        return {"version": response[0], "bytes_in_frame": response[1], "part_number": response[-16:]}

    def _initialize_encolink(self):
        if self.trigger_enabled:
            self.disable_trigger()
            self._trigger_last_state = True

        # Initialize EncoLink
        response = self.set_communication_protocol("SPI_EncoLink")
        if response != "SPI_ENCOLINK_MODE":
            raise ValueError("SPI EncoLink not established!")

        self.initialize_encolink_library()

    def _deinitialize_encolink(self):
        self.set_communication_protocol("SPI")
        if self._trigger_last_state:
            self.enable_trigger_master()
            self._trigger_last_state = False

    def write_registers(self, value: int, bank: int, address: int | str, length: int, is_signed: bool = False):
        # Initialize EncoLink
        self._initialize_encolink()

        # Write to register
        self._write_register(value, address, length)

        # Set back to SPI
        self._deinitialize_encolink()

    def read_registers(self, bank: int, address: int | str, length: int, is_signed: bool = False):
        # Initialize EncoLink
        self._initialize_encolink()

        # Read register
        data = self._read_register(address, length, is_signed)

        # Set back to SPI
        self._deinitialize_encolink()

        return data

    def _write_register(self, value: int, address: int, length: int):
        status = self.execute_command_with_response(
            f"W:{self._to_hex(length, 4)}:{self._to_hex(address, 8)}:{self._to_hex(value, 8)}"
        )

        self._raise_on_status(status)

    def _read_register(self, address: int | str, length: int, is_signed: bool):
        resp = self.execute_command_with_response(f"R:{self._to_hex(length, 4)}:{self._to_hex(address, 8)}")

        # Parse response -> e.g.: "0x09:0000000000"
        data = resp.split(":")[1]
        if data.startswith("0x"):
            data = data[2:]

        data = bytes.fromhex(data)
        return data

    @staticmethod
    def _to_hex(address: int, num_characters: int) -> str:
        return f"{address:0{num_characters}X}"
