from e201_test_platform.e201_acquisition.e201 import E201


class E201Q(E201):

    version = "E201Q"

    def __init__(self):
        super().__init__(self.version)

    def read_position(self):
        self.send(">")
        for line in self.read_lines():
            return int(line, 16)


if __name__ == "__main__":
    e201 = E201Q()
    e201.read_position()
