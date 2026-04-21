import time
import struct

from letp_e2019.e2019 import E2019
from letp_e2019.versions.e2019b import E2019B
from letp_e2019.versions.e2019p import E2019P
from letp_e2019.versions.e2019q import E2019Q
from letp_e2019.versions.e2019s import E2019S


class E2019Synced:
    _e201_versions = {
        "E2019B": E2019B,
        "E2019Q": E2019Q,
        "E2019P": E2019P,
        "E2019S": E2019S,
    }

    _required_top_keys = {"e201_dut", "e201_ref"}
    _required_device_keys = {"type", "comport"}

    def __init__(self, config: dict):
        self._validate_config(config)

        self.dut: E2019
        self.ref: E2019 | None = None  # type: ignore

        self.dut = self._init_device(config["e201_dut"], role="DUT")
        self.ref = self._init_device(config["e201_ref"], role="REF")

    def _validate_config(self, config: dict) -> None:
        """Config dictionary validation"""
        missing = self._required_top_keys - config.keys()
        if missing:
            raise KeyError(f"Missing required config keys: {missing}")

        for key in self._required_top_keys:
            device_cfg = config[key]
            if not isinstance(device_cfg, dict):
                raise TypeError(f"{key} must be a dict")

            missing_dev = self._required_device_keys - device_cfg.keys()
            if missing_dev:
                raise KeyError(f"{key} missing required keys: {missing_dev}")

    def _init_device(self, device_cfg: dict, role: str) -> E2019:
        """Device initialization"""
        comport = device_cfg["comport"]
        dev_type = device_cfg["type"]

        if comport is None:
            raise ValueError("Comport no selected!")

        if dev_type not in self._e201_versions:
            raise ValueError(f"{role}: Invalid E201 type '{dev_type}'. Available: {list(self._e201_versions.keys())}")

        try:
            cls = self._e201_versions[dev_type]
            instance = cls(comport)
            instance.open()
            return instance
        except Exception as e:
            raise ConnectionError(f"Cannot connect to E201 {role}! {e}") from e

    def read_position(self) -> bytearray:
        """Read synced position from master and slave (dut/ref)"""
        # determine sizes
        dut_len = self.dut.bytes if self.dut is not None else 8
        ref_len = 4
        ts_len = 8

        total_len = dut_len + ref_len + ts_len
        frame = bytearray(total_len)

        # DUT
        if self.dut is not None:
            raw = self._parse_master_raw(self.dut.read_position())

            # ensure correct size
            frame[0:dut_len] = raw[:dut_len].ljust(dut_len, b"\x00")
        else:
            frame[0:dut_len] = b"\x00" * dut_len

        # REF (after DUT)
        ref_offset = dut_len
        if self.ref is not None:
            raw = self._parse_reference_raw(self.ref.read_position())
            frame[ref_offset : ref_offset + ref_len] = raw[:ref_len]
        else:
            frame[ref_offset : ref_offset + ref_len] = b"\x00" * ref_len

        # TIMESTAMP (after REF)
        ts_offset = dut_len + ref_len
        ts = time.perf_counter_ns()
        struct.pack_into("<Q", frame, ts_offset, ts)

        return frame

    @staticmethod
    def _parse_master_raw(response: str):
        return bytes.fromhex(response.strip())

    @staticmethod
    def _parse_reference_raw(response: str):
        ref_raw = response.strip()
        # Trigger mode: T=XXXXXXXX
        if ref_raw.startswith("T="):
            hex_data = ref_raw.split("=")[1]

            if len(hex_data) != 8:
                raise ValueError(f"Invalid trigger frame: {ref_raw}")

            return bytes.fromhex(hex_data)  # 4 bytes

        # '>' command: 24 hex chars
        if len(ref_raw) != 24:
            raise ValueError(f"Invalid '>' frame length: {ref_raw}")

        full_bytes = bytes.fromhex(ref_raw)  # 12 bytes

        # Keep ONLY position (first 4 bytes)
        return full_bytes[:4]

    def enable_synced_sampling(self):
        """Enable synced sampling. Dut = master, ref = slave"""
        if self.dut is None or self.ref is None:
            return

        self.ref.enable_trigger_slave()
        self.dut.enable_trigger_master()

    def disable_synced_sampling(self):
        """Disable synced sampling"""
        if self.dut is not None:
            self.dut.disable_trigger()

        if self.ref is not None:
            self.ref.disable_trigger()

    def close_dut(self):
        """Close connection with dut"""
        if self.dut is not None:
            self.dut.disable_trigger()
            self.dut.close()
            self.dut = None

    def close_ref(self):
        """Close connection with ref"""
        if self.ref is not None:
            self.ref.disable_trigger()
            self.ref.close()
            self.ref = None

    def close_connection(self):
        """Close connection with dut and ref"""
        self.close_dut()
        self.close_ref()
