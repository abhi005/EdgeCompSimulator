import random
from time import sleep
from src.connections import MEC_Connection
from src.nodes import MEC, UE, Cloud
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from src.task import Task

class Env:
    def __init__(self, max_x, max_y) -> None:
        self.ue = None
        self.mec_servers = []
        self.cloud_server = None
        self.max_x = max_x
        self.max_y = max_y

    def add_mec_server(self, c_edge):
        self.mec_servers.append(MEC(self, c_edge))
        print('added a MEC server with c_edge: {}'.format(c_edge))

    def add_ue(self, c_local, p_send, p_calc, bandwidth, snr):
        self.ue = UE(self, c_local, p_calc, p_send)
        print('added a UE with c_local: {}, p_send: {} & p_calc: {}'.format(c_local, p_send, p_calc))
        for mec_server in self.mec_servers:
            SNR = random.randint(int(snr['min']), int(snr['max']))
            self.ue.add_mec_conn(MEC_Connection(bandwidth, SNR, mec_server))
            print('created a connection between UE and MEC server with bandwidth: {} & SNR: {}'.format(bandwidth, SNR))

    def add_cloud_server(self, c_cloud, tw):
        self.cloud = Cloud(self, c_cloud, tw)
        print('added a cloud server with c_cloud: {} & tw: {}'.format(c_cloud, tw))

    def set_task_vars(self, task_count, data_size, cycles, decision_period):
        self.task_count = task_count
        self.min_task_data_size_req = int(data_size['min'])
        self.max_task_data_size_req = int(data_size['max'])
        self.min_task_cycles_req = int(cycles['min'])
        self.max_task_cycles_req = int(cycles['max'])
        self.T = decision_period # in secs
        print('variables:')
        print('task data size, min: {}, max: {}'.format(self.min_task_data_size_req, self.max_task_data_size_req))
        print('task cycles, min: {}, max: {}'.format(
            self.min_task_cycles_req, self.max_task_cycles_req))
        print('decision period: {} secs'.format(self.T))

    def start(self):
        scheduler = BlockingScheduler(daemon=True)
        scheduler.add_job(self.task_generator, 'interval', seconds=self.T)
        scheduler.start()
    
    def task_generator(self):
        print('generating new tasks')
        for _ in range(self.task_count):
            size = random.randint(self.min_task_data_size_req, self.max_task_data_size_req)
            cycles = random.randint(self.min_task_cycles_req, self.max_task_cycles_req)
            self.ue.add_task(Task(size, cycles, self.ue))
            print('added a task to ue with data size req: {} & cpu cycles req: {}'.format(size, cycles))
