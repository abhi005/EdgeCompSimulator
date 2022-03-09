import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
        self.calculate_local_computing_delay()
        self.calculate_local_energy_consumption()
        self.calculate_edge_transmission_delay()
        self.calculate_cloud_transmission_delay()
        self.calculate_cloud_computing_delay()
        self.calculate_edge_transmission_energy()
        self.calculate_cloud_transmission_energy()
        self.calculate_edge_computing_delay()

    def calculate_local_computing_delay(self):
        self.T_calc_local = self.cycles / self.UE.comp_cap

    def calculate_local_energy_consumption(self):
        self.E_calc_local = self.T_calc_local * self.UE.p_calc

    def calculate_edge_transmission_delay(self):
        self.T_trans_edge = []
        for i in range(len(self.MECs)):
            self.T_trans_edge.append(self.size / self.MECs[i].data_rate)

    def calculate_cloud_transmission_delay(self):
        self.T_trans_cloud = min(self.T_trans_edge) + self.UE.env.cloud.tw

    def calculate_cloud_computing_delay(self):
        self.T_calc_cloud = self.cycles / self.UE.env.cloud.comp_cap

    def calculate_edge_transmission_energy(self):
        self.E_trans_edge = []
        for i in range(len(self.MECs)):
            self.E_trans_edge.append(self.T_trans_edge[i] * self.UE.p_send)

    def calculate_cloud_transmission_energy(self):
        self.E_trans_cloud = min(self.E_trans_edge)
    
    def calculate_edge_computing_delay(self):
        self.T_comp_edge = []
        for i in range(len(self.MECs)):
            self.T_comp_edge.append(self.cycles / self.MECs[i].mec_server.comp_cap)


