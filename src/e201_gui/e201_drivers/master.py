import time
from dataclasses import dataclass
from letp_e2019.e2019_synced import E2019Synced
from letp_e2019.e2019 import E2019


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
        self._validate_config(config)

        self.dut: E2019 | None = None  # type: ignore
        self.ref: E2019 | None = None

        self.dut = self._init_device(config["e201_dut"], role="DUT")
        self.ref = self._init_device(config["e201_ref"], role="REF")

        self.register_access: dict[str, RegisterAccessParameters] = {}

    def _init_device(self, device_cfg: dict, role: str) -> E2019:
        """Device initialization"""
        comport = device_cfg["comport"]
        dev_type = device_cfg["type"]

        if comport is None:
            return None  # type: ignore

        if dev_type not in self._e201_versions:
            raise ValueError(f"{role}: Invalid E201 type '{dev_type}'. Available: {list(self._e201_versions.keys())}")

        try:
            cls = self._e201_versions[dev_type]
            instance = cls(comport)
            instance.open()
            return instance
        except Exception as e:
            raise ConnectionError(f"Cannot connect to E201 {role}! {e}") from e

    def initialize_device(self, config: dict):
        self.dut.power_on(config.get("voltage", 5000))
        if self.dut.__class__.__name__ == "E2019P":
            self.dut.set_communication_protocol("SPI")  # type: ignore
            self.dut.set_clock_settings(config.get("polarity"), config.get("phase"))  # type: ignore

        if self.dut.__class__.__name__ in ("E2019P", "E2019B"):
            self.dut.set_clock_frequency(config.get("frequency"))  # type: ignore

        if self.dut.__class__.__name__ == "E2019S":
            self.dut.set_read_command(config.get("communication").lower())
            frame_length = (
                config.get("singleturn_bits", 0) + config.get("multiturn_bits", 0) + config.get("status_bits", 0)
            )
            self.dut.set_word_width(frame_length)

    def set_power(self, voltage):
        self.dut.power_on(voltage_mv=voltage)

    def check_framerate(self, n: int = 1000):
        start_time = time.perf_counter()
        for i in range(n):
            self.read_position()
        evaluation = time.perf_counter() - start_time
        framerate = n / evaluation
        return framerate

    def set_register_access(self, registers: dict):
        self.register_access = self._load_reg_acc_params(registers)

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

    def _get_reg_acc_params(self, parameter: str) -> RegisterAccessParameters:
        return self.register_access[parameter]

    def write_registers(self, value: int, bank: int, address: int | str, length: int, is_signed: bool):
        self.dut.write_registers(value, bank, address, length, is_signed)  # type: ignore

    def read_registers(self, bank: int, address: int | str, length: int, is_signed: bool):
        response = self.dut.read_registers(bank, address, length, is_signed)  # type: ignore
        return self.parse_response(response=response, length=length, is_signed=is_signed)

    def read_registers_params(self, parameter: str) -> dict:
        reg: RegisterAccessParameters = self._get_reg_acc_params(parameter)
        return self.read_registers(reg.bank, reg.address, reg.length, reg.is_signed)

    def write_registers_params(self, value: int, parameter: str):
        reg: RegisterAccessParameters = self._get_reg_acc_params(parameter)
        self.write_registers(
            value=value,
            bank=reg.bank,
            address=reg.address,
            length=reg.length,
            is_signed=reg.is_signed,
        )

    def set_multiturn(self, mt_cnt):
        """Set multiturn in encoder"""
        self.write_registers_params(mt_cnt, "multiturn_set")
        time.sleep(0.1)
        self._write_key()
        reg = self._get_reg_acc_params("multiturn_apply")
        self.write_registers_params(reg.value, "multiturn_apply")
        time.sleep(0.1)
        print(self.read_registers_params("multiturn_set"))

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

    def save_to_flash(self):
        """Save to non-volatile memory"""
        self._write_key()
        reg = self._get_reg_acc_params("save_to_flash")
        self.write_registers_params(reg.value, "save_to_flash")
        time.sleep(0.1)

    def factory_reset(self):
        """Factory reset of encoder"""
        self._write_key()
        reg = self._get_reg_acc_params("factory_reset")
        self.write_registers_params(reg.value, "factory_reset")
        time.sleep(0.1)

    def _write_key(self):
        if self.dut.__class__.__name__ == "E2019B":
            self.dut.write_register(0xCD, 0x48)
        pass

    @staticmethod
    def parse_response(response: bytes, length: int, is_signed: bool) -> dict:
        response_int = int.from_bytes(response, byteorder="big", signed=is_signed)
        response_str = format(response_int, f"0{length * 8}b")
        return {"response_raw": response, "response_int": response_int, "response_str": response_str}
