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
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class MultiUeEnv:
    def __init__(self, config) -> None:
        self.config = config
        self.max_x = int(self.config['length'])
        self.max_y = int(self.config['height'])
        self.T = int(self.config["decision_period"])
        self.mec_servers = []
        self.cloud = None
        self.ues = []
        self.metrics = Metrics()
        self.actor_lr = 0.0
        self.critic_lr = 0.0
        self.discount = 0.0
        print("decision_period: {}".format(self.T))
        self.mode = "test"
        self.load_test_config()

        self.load_task_config()
        self.load_cloud_config()
        self.load_mec_servers_config()
        self.load_ue_config()

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
        count = config["count"]
        self.snr = config["snr"]
        for _ in range(count):
            self.ues.append(
                UE(self, config["comp_capacity"], config["comp_power"], config["trans_power"], _))
            print("added a UE with c_local: {}, p_send: {} & p_calc: {}".format(
                config["comp_capacity"], config["trans_power"], config["comp_power"]))
            for mec_server in self.mec_servers:
                SNR = random.randint(
                    int(self.snr['min']), int(self.snr['max']))
                self.ues[_].add_mec_conn(MEC_Connection(
                    config["bandwidth"], SNR, mec_server))
                print("created a connection between UE {} and MEC server with bandwidth: {} & SNR: {}".format(
                    _, config["bandwidth"], SNR))

    def display_metrics(self):
        # for k in self.metrics.task_delays_per_episode:
        #     print("episode: {}, average task delay: {}".format(
        #         k, np.mean(self.metrics.task_delays_per_episode[k])))
        # for k in self.metrics.energy_consum_per_episode:
        #     print("episode: {}, average energy consumption: {}".format(
        #         k, np.mean(self.metrics.energy_consum_per_episode[k])))
        # for k in self.metrics.rewards_per_episode:
        #     print("episode: {}, average rewards: {}".format(
        #         k, np.mean(self.metrics.rewards_per_episode[k])))
        if self.mode == "train":
            with open(self.model_path + "training_rewards_per_episode.pkl", "wb") as f:
                pickle.dump(self.metrics.rewards_per_episode, f)
        else:
            with open(self.model_path + "multi_ue_testing_rewards_per_episode.pkl", "wb") as f:
                pickle.dump(self.metrics.rewards_per_episode, f)
            with open(self.model_path + "multi_ue_testing_energy_conumption.pkl", "wb") as f:
                pickle.dump(self.metrics.energy_consum_per_episode, f)
            with open(self.model_path + "multi_ue_testing_task_delays.pkl", "wb") as f:
                pickle.dump(self.metrics.task_delays_per_episode, f)
        exit(0)

    def start(self):
        self.scheduler = BlockingScheduler(daemon=True)
        self.scheduler.add_job(self.task_generator, "interval", seconds=self.T)
        self.scheduler.start()

    def task_generator(self):
        if self.max_episodes > 0:
            ue_indexs = [*range(len(self.ues))]
            random.shuffle(ue_indexs)
            for _ in ue_indexs:
                print("generating new tasks for ue {}".format(_))
                for i in range(self.max_task_count):
                    size = random.randint(
                        self.min_task_data_size_req, self.max_task_data_size_req)
                    cycles = random.randint(
                        self.min_task_cycles_req, self.max_task_cycles_req)
                    self.ues[_].add_task(Task(size, cycles, self.ues[_]))
                    print("added a task to ue {} with data size req: {} & cpu cycles req: {}".format(_,
                        size, cycles))
                reward = self.ues[_].process()
                print("reward for episode {} for ue {}: {}".format(self.max_episodes, _, reward))
            self.max_episodes -= 1
        else:
            self.display_metrics()
            self.scheduler.shutdown()
            exit(0)
