from os.path import exists
from src.metrics import Metrics
from src.task import Task
from apscheduler.schedulers.blocking import BlockingScheduler
from src.nodes import MEC, UE, Cloud
from src.connections import MEC_Connection
from time import sleep
import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Env:
    def __init__(self, config) -> None:
        self.config = config
        self.max_x = int(self.config['length'])
        self.max_y = int(self.config['height'])
        self.T = int(self.config["decision_period"])
        self.mec_servers = []
        self.cloud = None
        self.UE = None
        self.metrics = Metrics()
        self.actor_lr = 0.0
        self.critic_lr = 0.0
        self.discount = 0.0
        print("decision_period: {}".format(self.T))

        if "train" in self.config:
            self.mode = "train"
            self.load_training_config()
        else:
            self.mode = "test"
            self.load_test_config()

        self.load_task_config()
        self.load_cloud_config()
        self.load_mec_servers_config()
        self.load_ue_config()

    def load_training_config(self):
        config = self.config["train"]
        self.model_path = config["save_path"]
        self.actor_lr = config["actor_lr"]
        self.critic_lr = config["critic_lr"]
        self.discount = config["discount"]
        self.max_episodes = config["max_episodes"]
        print("model_save_path: {}\nactor_lr: {}\ncritic_lr: {}\ndiscount: {}\nmax_episodes: {}".format(
            self.model_path, self.actor_lr, self.critic_lr, self.discount, self.max_episodes))

    def load_test_config(self):
        config = self.config["test"]
        self.model_path = config["model_path"]
        self.max_episodes = config["runs"]

    def load_task_config(self):
        config = self.config["tasks"]
        self.time_impo = config["time_impo"]
        self.energy_impo = 1.0 - self.time_impo
        self.max_task_count = config["count"]
        self.min_task_data_size_req = int(config["data_size"]["min"])
        self.max_task_data_size_req = int(config["data_size"]["max"])
        self.min_task_cycles_req = int(config["cycles"]["min"])
        self.max_task_cycles_req = int(config["cycles"]["max"])
        print("max_task_count: {}\nmin_task_data_req: {}\nmax_task_data_size_req: {}\nmin_task_cycles_req: {}\nmax_task_cycles_req: {}\ntime_impo: {}\nenergy_impo: {}".format(
            self.max_task_count, self.min_task_data_size_req, self.max_task_data_size_req, self.min_task_cycles_req, self.max_task_cycles_req, self.time_impo, self.energy_impo))

    def load_cloud_config(self):
        config = self.config["cloud_server"]
        self.cloud = Cloud(
            self, config["comp_capacity"], config["trans_delay"])
        print("added a cloud server with c_cloud: {} & tw: {}".format(
            config["comp_capacity"], config["trans_delay"]))

    def load_mec_servers_config(self):
        config = self.config["mec_servers"]
        count = config["count"]
        for _ in range(count):
            self.add_mec_server(config["comp_capacity"])

    def add_mec_server(self, c_edge):
        self.mec_servers.append(MEC(self, c_edge))
        print('added a MEC server with c_edge: {}'.format(c_edge))

    def load_ue_config(self):
        config = self.config["ue"]
        self.ue = UE(self, config["comp_capacity"],
                     config["comp_power"], config["trans_power"])
        self.snr = config["snr"]
        print("added a UE with c_local: {}, p_send: {} & p_calc: {}".format(
            config["comp_capacity"], config["trans_power"], config["comp_power"]))
        for mec_server in self.mec_servers:
            SNR = random.randint(int(self.snr['min']), int(self.snr['max']))
            self.ue.add_mec_conn(MEC_Connection(
                config["bandwidth"], SNR, mec_server))
            print("created a connection between UE and MEC server with bandwidth: {} & SNR: {}".format(
                config["bandwidth"], SNR))

    def display_metrics(self):
        for k in self.metrics.task_delays_per_episode:
            print("episode: {}, average task delay: {}".format(
                k, np.mean(self.metrics.task_delays_per_episode[k])))
        for k in self.metrics.energy_consum_per_episode:
            print("episode: {}, average energy consumption: {}".format(k, np.mean(self.metrics.energy_consum_per_episode[k])))
        for k in self.metrics.rewards_per_episode:
            print("episode: {}, average rewards: {}".format(
                k, np.mean(self.metrics.rewards_per_episode[k])))
        exit(0)

    def start(self):
        self.scheduler = BlockingScheduler(daemon=True)
        self.scheduler.add_job(self.task_generator, "interval", seconds=self.T)
        self.scheduler.start()

    def task_generator(self):
        if self.max_episodes > 0:
            print("generating new tasks")
            for _ in range(self.max_task_count):
                size = random.randint(
                    self.min_task_data_size_req, self.max_task_data_size_req)
                cycles = random.randint(
                    self.min_task_cycles_req, self.max_task_cycles_req)
                self.ue.add_task(Task(size, cycles, self.ue))
                print("added a task to ue with data size req: {} & cpu cycles req: {}".format(
                    size, cycles))
            reward = self.ue.process()
            print("reward for episode {}: {}".format(self.max_episodes, reward))
            self.max_episodes -= 1
        else:
            self.display_metrics()
            self.scheduler.shutdown()
            exit(0)
