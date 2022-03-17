import numpy as np

class Metrics():
    def __init__(self) -> None:
        self.task_delays_per_episode = {}
        self.energy_consum_per_episode = {}
        self.rewards_per_episode = {}

    def add_task_delay(self, episode, delay):
        if not episode in self.task_delays_per_episode:
            self.task_delays_per_episode[episode] = []
        self.task_delays_per_episode[episode].append(delay)

    def add_task_energy_consum(self, episode, energy):
        if not episode in self.energy_consum_per_episode:
            self.energy_consum_per_episode[episode] = []
        self.energy_consum_per_episode[episode].append(energy)

    def add_episode_reward(self, episode, reward):
        self.rewards_per_episode[episode] = reward