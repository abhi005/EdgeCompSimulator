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
        self.cloud = None
        self.tasks = []
        # CPU power consumption in watts
        self.p_calc = p_calc
        # transmission power in Hz
        self.p_send = p_send
        # agent

    def add_mec_conn(self, conn):
        self.mec_conns.append(conn)

    def add_task(self, task):
        self.tasks.append(task)

class MEC(Node):
    def __init__(self, env, comp_cap) -> None:
        super().__init__(env, comp_cap)
        self.tasks = []

class Cloud(Node):
    def __init__(self, env, comp_cap, tw) -> None:
        super().__init__(env, comp_cap)
        self.tasks = []
        # constant transmission delay b/w all MEC server and cloud server
        self.tw = tw

