
from e201_test_platform.e201_acquisition.e201 import E201

class E201B(E201):

    version = "E201Q"

    def __init__(self):
        super().__init__(self.version)

    def read_position(self):
        self.send(">")
        for line in self.read_lines():
            frame = int(line, 16)
            # Extract position bits here
            return frame



if __name__ == "__main__":
    e201 = E201B()
    e201.read_position()