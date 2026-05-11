import time
from dataclasses import dataclass
from letp_e2019.e2019_synced import E2019Synced


@dataclass
class RegisterAccessParameters:
    value: int
    address: int
    execute: bool
    bank: int = 0
    length: int = 1
    is_signed: bool = False

    def __post_init__(self):
        if isinstance(self.address, str):
            self.address = int(self.address, 16)

        if isinstance(self.is_signed, int):
            self.is_signed = bool(self.is_signed)


class Master(E2019Synced):
    def __init__(self, config: dict):
        super().__init__(config)
        self.register_access = self._load_reg_acc_params({})

    def initialize_device(self, config: dict):
        self.set_power(config.get("voltage", 5000))
        if self.dut.__class__.__name__ == "E2019P":
            self.dut.set_communication_protocol("SPI")  # type: ignore
            self.dut.set_clock_settings(config.get("polarity"), config.get("phase"))  # type: ignore

        if self.dut.__class__.__name__ in ("E2019P", "E2019B"):
            self.dut.set_clock_frequency(config.get("frequency"))  # type: ignore

        if self.dut.__class__.__name__ == "E2019S":
            self.dut.set_read_command(config["communication"].lower())
            frame_length = (
                config.get("singleturn_bits", 0) + config.get("multiturn_bits", 0) + config.get("status_bits", 0)
            )
            self.dut.set_word_width(frame_length)  # type: ignore

    def set_register_access(self, params: dict):
        self.register_access = self._load_reg_acc_params(params)

    def set_power(self, voltage):
        self.dut.power_on(voltage_mv=voltage)

    def check_framerate(self, n: int = 1000):
        start_time = time.perf_counter()
        for i in range(n):
            self.read_position()
        evaluation = time.perf_counter() - start_time
        framerate = n / evaluation
        return framerate

    def read_registers(self, bank, address, length, is_signed) -> dict:
        """
        Read multiple registers.

        Args:
            bank (int): Register bank (default: 0)
            address (int): Start address register (0-127)
            length (int): Number of registers to read (0-64)
            is_signed (bool): Parameter type - signed/unsigned

        Returns:
            tuple[list[int], int, str]: A tuple containing:
                - list[int]: Raw list of registers (integer values).
                - int: Integer value of merged bits.
                - str: Response in string format (all bits).
        """
        response = self.dut.read_registers(bank, address, length, is_signed)
        return self.parse_response(response, length, is_signed)

    def write_registers(self, value, bank, address, length, is_signed):
        """
        Write to multiple registers.
        Args:
            bank (int): Register bank (default: 0)
            address (int): Start address register (0-127)
            length (int): Number of registers to read (0-64)
            value (int):  Passed value to write
            is_signed (bool): Parameter type - signed/unsigned

        Returns:
            None
        """
        self.dut.write_registers(value, bank, address, length, is_signed)

    @staticmethod
    def _load_reg_acc_params(parameters) -> dict[str, RegisterAccessParameters]:
        reg_acc = {}
        for reg_name, values in parameters.items():
            reg_acc[reg_name] = RegisterAccessParameters(
                value=values.get("value"),
                bank=values.get("bank"),
                address=values.get("address"),
                length=values.get("length"),
                is_signed=values.get("is_signed"),
                execute=values.get("execute"),
            )

        return reg_acc

    def get_reg_acc_params(self, parameter: str) -> RegisterAccessParameters:
        return self.register_access[parameter]

    def read_registers_params(self, parameter: str) -> dict:
        reg = self.get_reg_acc_params(parameter)
        return self.read_registers(reg.bank, reg.address, reg.length, reg.is_signed)

    def write_registers_params(self, value: int, parameter: str):
        reg = self.get_reg_acc_params(parameter)
        self.write_registers(
            value=value,
            bank=reg.bank,
            address=reg.address,
            length=reg.length,
            is_signed=reg.is_signed,
        )

    def set_multiturn(self, multiturn_value):
        """Set multiturn in encoder"""
        self.write_registers_params(multiturn_value, "multiturn_set")
        time.sleep(0.1)
        self._write_key()
        reg = self.get_reg_acc_params("multiturn_apply")
        self.write_registers_params(reg.value, "multiturn_apply")
        time.sleep(0.1)

    def set_position_offset(self, offset):
        """
        Set position offset requires following sequence: \n
        - write offset 0 \n
        - read current position in counts \n
        - write offset of current position  \n
        - perform power cycle \n
        - check current position after offset set \n
        :param offset: offset to set in counts
        :return: position write status
        """
        # Write offset
        self.write_registers_params(offset, "position_offset")

        # Read offset
        response_int = self.read_registers_params("position_offset")["response_int"]

        if response_int != offset:
            raise ValueError("Error writing position offset!")

        self.save_to_flash()

    def set_counting_direction(self, direction, offset=0, connector=1):
        """
        Set counting direction requires following sequence: \n
        - write direction \n
        - set position offset \n
        - perform power cycle  \n
        - perform power cycle \n
        - check current position after offset set \n
        :param direction: direction to set
        :param offset: offset to set in counts
        :return: position write status
        """
        # Write direction
        self.write_registers_params(direction, "counting_direction")
        time.sleep(0.1)
        self.save_to_flash()

        # Set position offset
        self.write_registers_params(offset, "position_offset")
        time.sleep(0.07)

        # Perform power cycle
        self.dut.power_off()
        time.sleep(0.3)
        self.dut.power_on()

        # Set position offset
        self.write_registers_params(offset, "position_offset")
        time.sleep(0.07)

        # Check counting direction register
        reg_list = self.read_registers_params("counting_direction")

        return reg_list["response_int"]

    def start_self_calibration(self):
        """Start self calibration on encoder"""
        self._write_key()
        try:
            reg = self.get_reg_acc_params("start_calibration")
            self.write_registers_params(reg.value, "start_calibration")
        except Exception:
            pass

    def wait_while_selfcal_is_active(self, timeout):
        """
        Encoder stays unresponsive during self calibration. Call this function to know when encoder is ready again.
        """
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                return self.read_self_calibration_finished()

            except Exception:  # DUT is not responsive while self calibration is in progress
                pass

    def read_self_calibration_finished(self):
        try:
            return self.read_registers_params("self_cal_increment")
        except Exception:
            return self.read_registers_params("self_cal_status")

    def save_to_flash(self):
        """Save to non-volatile memory"""
        self._write_key()
        reg = self.get_reg_acc_params("save_to_flash")
        self.write_registers_params(reg.value, "save_to_flash")
        time.sleep(0.1)

    def factory_reset(self):
        """Factory reset of encoder"""
        self._write_key()
        reg = self.get_reg_acc_params("factory_reset")
        self.write_registers_params(reg.value, "factory_reset")
        time.sleep(0.1)

    def _write_key(self):
        if self.dut.__class__.__name__ == "E2019B":
            self.dut._write_register(0xCD, 0x48)  # type: ignore
        return

    @staticmethod
    def parse_response(response: bytes, length: int, is_signed: bool) -> dict:
        response_int = int.from_bytes(response, byteorder="big", signed=is_signed)
        response_str = format(response_int, f"0{length * 8}b")
        return {"response_raw": response, "response_int": response_int, "response_str": response_str}

    def read_miss_image(self) -> dict | None:
        """
        Read miss image.
        Returns:
            56 MIS image values
            Decoded absolute position code 1
            Decoded absolute position code 2
            Decoding quadrant (first sensor index)
        """
        try:
            self._write_key()
            if self.dut.__class__.__name__ == "E2019B":
                return self._read_miss_image_biss()
            elif self.dut.__class__.__name__ == "E2019P":
                return self._read_miss_image_encolink()
            else:
                return None

        except Exception:
            return None

    def _read_miss_image_biss(self):
        self.write_registers(ord("G"), 0, 0x49, 1, False)
        response = self.read_registers(223, 0x00, 64, False)
        mis_image_values = [int.from_bytes([point], "big", signed=True) * 16 for point in response["response_raw"][:56]]
        abs_pos_code_1 = format(
            int.from_bytes(response["response_raw"][56:58], byteorder="big", signed=False), f"0{2 * 8}b"
        )
        abs_pos_code_2 = format(
            int.from_bytes(response["response_raw"][58:60], byteorder="big", signed=False), f"0{2 * 8}b"
        )
        first_sensor_index = response["response_raw"][63]

        return {
            "mis_image_values": mis_image_values,
            "abs_pos_code_1": abs_pos_code_1,
            "abs_pos_code_2": abs_pos_code_2,
            "first_sensor_index": first_sensor_index,
        }

    def _read_miss_image_encolink(self):
        self.write_registers(ord("G"), 0, 0xBD, 1, False)
        raw_values = self._read_raw_encolink_mis_values()
        mis_image_values = [int.from_bytes(raw_values[i : i + 2], "big", signed=True) for i in range(0, 112, 2)]
        decoded_values = self.read_registers(0, 0x20084, 4, False)
        abs_pos_code_1 = format(
            int.from_bytes(decoded_values["response_raw"][:2], byteorder="big", signed=False), f"0{2 * 8}b"
        )
        abs_pos_code_2 = format(
            int.from_bytes(decoded_values["response_raw"][2:4], byteorder="big", signed=False), f"0{2 * 8}b"
        )
        decoded_values = self.read_registers(0, 0x2008A, 1, False)
        first_sensor_index = decoded_values["response_int"]

        return {
            "mis_image_values": mis_image_values,
            "abs_pos_code_1": abs_pos_code_1,
            "abs_pos_code_2": abs_pos_code_2,
            "first_sensor_index": first_sensor_index,
        }

    def _read_raw_encolink_mis_values(self):
        base = 0x20000
        miss_image = []
        for _ in range(112):
            miss_image.append(self.read_registers(0, base, 4, False)["response_raw"])
            base += 4

        return b"".join(miss_image)
