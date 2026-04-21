import numpy as np
from PyQt5 import QtCore


class PositionBuffer:
    def __init__(self, size=20000):
        self.size = size
        self.index = 0
        self.count = 0

        # raw
        self.ref_counts = np.zeros(size, dtype=np.uint32)
        self.dut_counts = np.zeros(size, dtype=np.int32)
        self.ts = np.zeros(size, dtype=np.float64)
        self.sample_idx = np.zeros(size, dtype=np.int64)

        # processed
        self.ref_deg = np.zeros(size, dtype=np.float64)
        self.dut_deg = np.zeros(size, dtype=np.float64)
        self.err_deg = np.zeros(size, dtype=np.float64)
        self.inl_deg = np.zeros(size, dtype=np.float64)
        self.dnl_deg = np.zeros(size, dtype=np.float64)
        self.noise = np.zeros(size, dtype=np.float64)
        self.multiturn = np.zeros(size, dtype=np.int64)

        self.lock = QtCore.QMutex()

    def append(
        self,
        sample_idx: int,
        ts: float,
        dut_counts: int,
        ref_counts: int,
        dut_deg: float,
        ref_deg: float,
        err_deg: float,
        inl_deg: float,
        dnl_deg: float,
        noise: float,
        multiturn: int | None,
    ):
        with QtCore.QMutexLocker(self.lock):
            i = self.index

            self.sample_idx[i] = sample_idx
            self.ts[i] = ts
            self.dut_counts[i] = dut_counts
            self.ref_counts[i] = ref_counts

            self.dut_deg[i] = dut_deg
            self.ref_deg[i] = ref_deg
            self.err_deg[i] = err_deg
            self.inl_deg[i] = inl_deg
            self.dnl_deg[i] = dnl_deg
            self.noise[i] = noise
            self.multiturn[i] = multiturn

            self.index = (i + 1) % self.size
            self.count = min(self.count + 1, self.size)

    def snapshot(self, n: int):
        with QtCore.QMutexLocker(self.lock):
            n = min(n, self.count)
            if n <= 0:
                return None

            idx = self.index
            size = self.size
            indices = np.arange(idx - n, idx) % size

            return {
                "sample_idx": self.sample_idx[indices].copy(),
                "ts": self.ts[indices].copy(),
                "dut_counts": self.dut_counts[indices].copy(),
                "ref_counts": self.ref_counts[indices].copy(),
                "dut_deg": self.dut_deg[indices].copy(),
                "ref_deg": self.ref_deg[indices].copy(),
                "err_deg": self.err_deg[indices].copy(),
                "inl_deg": self.inl_deg[indices].copy(),
                "dnl_deg": self.dnl_deg[indices].copy(),
                "noise": self.noise[indices].copy(),
                "multiturn": self.multiturn[indices].copy(),
            }
