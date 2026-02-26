import serial
import threading
import time


class E201Device:
    def __init__(self, port: str, command: str = ">"):
        self.ser = serial.Serial(
            port,
            baudrate=115200,
            timeout=0,
            write_timeout=0
        )

        self.command = command
        self._buffer = bytearray()

        self._lock = threading.Lock()
        self._latest = None  # (timestamp, position)

        self._running = False
        self._thread = None

    # -----------------------------
    # Public API
    # -----------------------------

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def latest(self):
        with self._lock:
            return self._latest

    # -----------------------------
    # Internal loop
    # -----------------------------

    def _loop(self):
        send = self.ser.write
        read = self.ser.read
        in_waiting = self.ser.in_waiting

        cmd = self.command.encode() + b"\r"

        while self._running:

            # Send position request
            send(cmd)

            # Read available bytes
            data = read(self.ser.in_waiting or 1)

            if data:
                self._buffer.extend(data)

                while b"\r\n" in self._buffer:
                    line, _, self._buffer = self._buffer.partition(b"\r\n")

                    try:
                        position = int(line, 16)
                    except ValueError:
                        continue

                    ts = time.perf_counter()

                    with self._lock:
                        self._latest = (ts, position)