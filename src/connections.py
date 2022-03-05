import math


class Connection:
    def __init__(self, B, snr) -> None:
        # bandwidth in Hz
        self.B = B
        self.snr = snr
        self.calculate_data_rate()

    def calculate_data_rate(self):
        self.data_rate = self.B * math.log(1.0 + self.snr, 10)

class MEC_Connection(Connection):
    def __init__(self, B, snr, mec_server) -> None:
        super().__init__(B, snr)
        self.mec_server = mec_server
        self.alloted_data_size = 0.0
        # Gigacycles/sec
        self.alloted_cpu_cycles = 0
