import numpy as np

class Metrics():
    def __init__(self) -> None:
        self.task_delays_per_episode = {}
        self.energy_consum_per_episode = {}
        self.rewards_per_episode = {}

    def add_task_delay(self, ue_id, episode, delay):
        if not ue_id in self.task_delays_per_episode:
            self.task_delays_per_episode[ue_id] = {}
        if not episode in self.task_delays_per_episode[ue_id]:
            self.task_delays_per_episode[ue_id][episode] = []
        self.task_delays_per_episode[ue_id][episode].append(delay)

    def add_task_energy_consum(self, ue_id, episode, energy):
        if not ue_id in self.energy_consum_per_episode:
            self.energy_consum_per_episode[ue_id] = {}
        if not episode in self.energy_consum_per_episode[ue_id]:
            self.energy_consum_per_episode[ue_id][episode] = []
        self.energy_consum_per_episode[ue_id][episode].append(energy)

    def add_episode_reward(self, ue_id, episode, reward):
        if not ue_id in self.rewards_per_episode:
            self.rewards_per_episode[ue_id] = {}
        self.rewards_per_episode[ue_id][episode] = reward