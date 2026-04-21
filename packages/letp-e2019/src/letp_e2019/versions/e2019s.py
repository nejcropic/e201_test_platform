from letp_e2019.e2019 import E2019Power


class E2019S(E2019Power):
    _available_communications = {"biss": "4", "ssi": ">"}
    available_freq = {
        35: 1,
        70: 2,
        140: 3,
        280: 4,
        560: 5,
        1100: 6,
        2200: 7,
        4400: 8,
    }

    def __init__(self, comm):
        super().__init__(comm)
        self.bytes = 4

    def set_read_command(self, communication: str):
        if communication not in self._available_communications:
            raise ValueError(f"Communication not in available communications! {self._available_communications.keys()}")

        self.read_command = self._available_communications[communication]
        self.bytes = 8 if communication.lower() == "biss" else 4

    def set_word_width(self, word_width: int):
        if self.read_command != ">":
            return

        command = f"B{word_width:02d}\r"
        self.execute_command_with_response(command)

        width_set = self.check_word_width().split(" ")[0]
        if word_width != int(width_set):
            raise ValueError(f"Word width not set! Word width: {width_set}")

    def check_word_width(self):
        return self.execute_command_with_response("b")
