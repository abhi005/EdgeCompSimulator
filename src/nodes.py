import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import uuid
import math
from datetime import datetime, timedelta
from apscheduler.triggers.date import DateTrigger
from apscheduler.schedulers.background import BackgroundScheduler
from src.utils import Queue
from src.agent2 import Agent
import numpy as np
from sre_parse import State
import random


class Node:
    def __init__(self, env, comp_cap) -> None:
        self.env = env
        self.x = random.randint(0, env.max_x)
        self.y = random.randint(0, env.max_y)
        # computation capacity in Gigacycles/sec
        self.comp_cap = comp_cap


class UE(Node):
    def __init__(self, env, comp_cap, p_calc, p_send) -> None:
        super().__init__(env, comp_cap)
        self.mec_conns = []
        self.tasks = Queue(self.env.max_task_count)
        # CPU power consumption in watts
        self.p_calc = p_calc
        # transmission power in Hz
        self.p_send = p_send
        self.agent = Agent(mode=env.mode,
            model_path=env.model_path,
            state_dim=self.get_state_dim(),
            action_dim=self.get_action_dim(),
            actor_lr=self.env.actor_lr,
            critic_lr=self.env.critic_lr,
            discount=self.env.discount)

    def get_action_dim(self):
        return (2 + len(self.env.mec_servers)) * self.env.max_task_count

    def get_state_dim(self):
        return (len(self.env.mec_servers) * 2) + (self.env.max_task_count * 2) + 2

    def add_mec_conn(self, conn):
        self.mec_conns.append(conn)
        self.calculate_state()
        self.calculate_high_state()
        self.calculate_low_state()

    def add_task(self, task):
        self.tasks.add(task)
        self.calculate_state()
        self.calculate_high_state()
        self.calculate_low_state()

    def poll_task(self):
        self.tasks.poll()
        self.calculate_state()
        self.calculate_high_state()
        self.calculate_low_state()

    def calculate_high_state(self):
        self.high_state = np.append(
            [self.env.snr['max']] * len(self.mec_conns), [self.env.max_task_count] * len(self.mec_conns))
        self.high_state = np.append(self.high_state, self.env.max_task_count)
        temp = [self.env.max_task_data_size_req] * self.env.max_task_count
        self.high_state = np.append(self.high_state, temp)
        temp = [self.env.max_task_cycles_req] * self.env.max_task_count
        self.high_state = np.append(self.high_state, temp)
        self.high_state = np.append(self.high_state, self.env.max_task_count)

    def calculate_low_state(self):
        self.low_state = np.append(
            [self.env.snr['min']] * len(self.mec_conns), [0] * len(self.mec_conns))
        self.low_state = np.append(self.low_state, 0)
        temp = [self.env.min_task_data_size_req] * self.env.max_task_count
        self.low_state = np.append(self.low_state, temp)
        temp = [self.env.min_task_cycles_req] * self.env.max_task_count
        self.low_state = np.append(self.low_state, temp)
        self.low_state = np.append(self.low_state, 0)

    def calculate_state(self):
        self.state = np.append([conn.snr for conn in self.mec_conns], [
                               len(conn.mec_server.tasks) for conn in self.mec_conns])
        self.state = np.append(self.state, self.tasks.get_len())
        temp = [task.size for task in self.tasks.list]
        temp = np.append(temp, np.zeros(self.env.max_task_count - len(temp)))
        self.state = np.append(self.state, temp)
        temp = [task.cycles for task in self.tasks.list]
        temp = np.append(temp, np.zeros(self.env.max_task_count - len(temp)))
        self.state = np.append(self.state, temp)
        self.state = np.append(self.state, self.tasks.get_len())

    def get_normalized_state(self):
        # print("state: {}".format(self.state))
        # print("shape: {}".format(len(self.state)))
        # print("high state: {}".format(self.high_state))
        # print("shape: {}".format(len(self.high_state)))
        # print("low state: {}".format(self.low_state))
        # print("shape: {}".format(len(self.low_state)))
        return self.state / (self.high_state - self.low_state)

    def yield_task_action(self, action, dim):
        for i in range(0, len(action), dim):
            yield action[i:i + dim]

    def take_step(self, action):
        single_task_dim = len(action) / self.env.max_task_count
        reward = 0.0
        decisions = []
        i = 0
        for a in self.yield_task_action(action, int(single_task_dim)):
            print("action probs:", a)
            decision = np.argmax(a)
            decisions.append(int(i * single_task_dim + decision))
            task = self.tasks.peek()
            if task != None:
                if decision == 0:
                    print("local computation")
                    # local computation
                    comp_delay = task.T_calc_local
                    energy_consum = task.E_calc_local
                    curr_reward = self.env.time_impo * comp_delay + \
                        self.env.energy_impo * energy_consum
                    reward += curr_reward
                    self.poll_task()
                    print("reward: {}".format(curr_reward))
                elif decision == len(a) - 1:
                    print("cloud offloading")
                    # cloud computation
                    trans_delay = task.T_trans_cloud
                    comp_delay = task.T_calc_cloud
                    energy_consum = task.E_trans_cloud
                    curr_reward = self.env.time_impo * \
                        (comp_delay + trans_delay) + \
                        self.env.energy_impo * energy_consum
                    reward += curr_reward
                    self.poll_task()
                    print("reward: {}".format(curr_reward))
                else:
                    print("edge offloading")
                    # MEC server computation
                    MEC_index = decision - 1
                    if len(self.mec_conns[MEC_index].mec_server.tasks) < self.env.max_task_count:
                        print("offloading to MEC server: {}".format(MEC_index))
                        trans_delay = task.T_trans_edge[MEC_index]
                        comp_delay = task.T_comp_edge[MEC_index]
                        energy_consum = task.E_trans_edge[MEC_index]
                        curr_reward = self.env.time_impo * \
                            (comp_delay + trans_delay) + \
                            self.env.energy_impo * energy_consum
                        reward += curr_reward
                        self.poll_task()
                        self.mec_conns[MEC_index].mec_server.add_task(task)
                        print("reward: {}".format(curr_reward))
                    else:
                        print("couldn't offload to MEC server, server overloaded")
                        reward -= 100
                        print("reward: {}".format(-100))
            i += 1

        reward = 100 / (10 + reward)
        return self.get_normalized_state(), reward, decisions

    def process(self):
        # decide where to offload the tasks using agent
        s = self.get_normalized_state()
        action = self.agent.get_action(s)
        s_, r, action = self.take_step(action)
        self.agent.learn(state=s, action=action, reward=r, state_=s_)
        self.agent.save()
        return r

class MEC(Node):
    def __init__(self, env, comp_cap) -> None:
        super().__init__(env, comp_cap)
        self.tasks = {}
        self.scheduler = BackgroundScheduler(daemon=True)
        self.scheduler.start()

    def add_task(self, task):
        id = str(uuid.uuid4())
        after_secs = datetime.now() + timedelta(seconds=int(math.ceil(task.cycles / self.comp_cap)))
        self.tasks[id] = task
        self.scheduler.add_job(self.complete_task, args=(
            id,), trigger=DateTrigger(after_secs), max_instances=1)

    def complete_task(self, id):
        print("removing task from MEC serve with id: {}".format(id))
        self.tasks.pop(id)


class Cloud(Node):
    def __init__(self, env, comp_cap, tw) -> None:
        super().__init__(env, comp_cap)
        # constant transmission delay b/w all MEC server and cloud server
        self.tw = tw
