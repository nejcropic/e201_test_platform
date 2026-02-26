from e201_test_platform.e201_acquisition.e201 import E201


class DualE201Acquisition:

    def __init__(self, ref_dev: E201, dut_dev: E201):
        self.ref = ref_dev
        self.dut = dut_dev

    def start(self):
        self.ref.start()
        self.dut.start()

    def stop(self):
        self.ref.stop()
        self.dut.stop()

    def sample_pair(self):
        ref_sample = self.ref.latest()
        dut_sample = self.dut.latest()

        if not ref_sample or not dut_sample:
            return None

        return {
            "ref_time": ref_sample[0],
            "ref_pos": ref_sample[1],
            "dut_time": dut_sample[0],
            "dut_pos": dut_sample[1],
            "time_skew": abs(ref_sample[0] - dut_sample[0])
        }

if __name__ == "__main__":
    from e201_test_platform.e201_acquisition.e201b import E201B
    from e201_test_platform.e201_acquisition.e201q import E201Q
    ref = E201Q()
    dut = E201B()

    acq = DualE201Acquisition(ref, dut)
    acq.start()

    try:
        while True:
            sample = acq.sample_pair()
            if sample:
                print(sample)
    except KeyboardInterrupt:
        acq.stop()