class Task:
    def __init__(self, size, cycles, UE) -> None:
        # data size requirement for task
        self.size = size
        # CPU cycles requirement for task
        self.cycles = cycles
        # curr alloted data size by MEC server to this task
        self.curr_alloted_size = 0.0
        # curr alloted CPU cycles by MEC server to this task
        self.curr_alloted_cpu_cycles = 0
        self.UE = UE
        self.MECs = UE.mec_conns
        self.cloud = UE.cloud
        # self.calculate_local_computing_delay()
        # self.calculate_local_energy_consumption()
        # self.calculate_edge_computing_delay()
        # self.calculate_cloud_computing_delay()

    def calculate_local_computing_delay(self):
        self.t_calc_local = self.cycles / self.UE.comp_cap

    def calculate_local_energy_consumption(self):
        self.local_energy_req = self.t_calc_local * self.UE.p_calc
    
    def calculate_edge_computing_delay(self):
        self.edge_comp_delays = []
        for i in range(len(self.MECs)):
            self.edge_comp_delays.append(self.cycles / self.MECs[i].alloted_cpu_cycles)

    def calculate_cloud_computing_delay(self):
        self.cloud_comp_delay = self.cycles / self.cloud.comp_cap

    def calculate_edge_transmission_delay(self):
        self.edge_trans_delays = []
        for i in range(len(self.MECs)):
            self.edge_trans_delays.append(self.MECs[i].alloted_data_size / self.MECs[i].data_rate)

    def calculate_cloud_transmission_delay(self):
        self.cloud_trans_delay = min(self.edge_trans_delays) + self.cloud.trans_delay

    def calculate_edge_transmission_energy(self):
        self.edge_trans_energies = []
        for i in range(len(self.MECs)):
            self.edge_trans_energies.append(self.edge_trans_delays[i] * self.MECs[i].p_send)

    def calculate_cloud_transmission_energy(self):
        self.cloud_trans_energy = min(self.edge_trans_energies)


