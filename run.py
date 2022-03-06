import yaml
import argparse
from src.env import Env
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def read_file(path):
    with open(path, mode='r') as file:
        data = file.read()
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='xml config file for simulator', type=str, action='store', required=True)
    args = parser.parse_args()

    # loading yaml config for simulator
    config = yaml.safe_load(read_file(args.config))['scene']
    print('yaml config loading done')
    
    mode = config['mode']
    if mode == "train":
        actor_lr = config['actor_lr']
        critic_lr = config['critic_lr']
        discount = config['discount']
    env = Env(int(config['length']), int(config['height']), mode, actor_lr, critic_lr, discount)

    # parsing task configs
    task_count = config['task_req']['count']
    data_size = config['task_req']['data_size']
    cycles = config['task_req']['cycles']
    decision_period = config['decision_period']
    env.set_task_vars(task_count, data_size, cycles, decision_period)

    # parsing mec server configs
    mec_server_count = int(config['mec_servers']['count'])
    c_edge = float(config['mec_servers']['comp_capacity'])
    for _ in range(mec_server_count):
        env.add_mec_server(c_edge)

    # parsing ue configs
    c_local = float(config['ues']['comp_capacity'])
    p_send = float(config['ues']['trans_power'])
    p_calc = float(config['ues']['comp_power'])
    bandwidth = float(config['ues']['bandwidth'])
    snr = config['ues']['snr']
    env.add_ue(c_local, p_send, p_calc, bandwidth, snr)

    # parsing cloud server configs
    c_cloud = float(config['cloud_server']['comp_capacity'])
    tw = float(config['cloud_server']['trans_delay'])
    env.add_cloud_server(c_cloud, tw)
    
    # starting the simulator
    env.start()


if __name__ == "__main__":
    main()