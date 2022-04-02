import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import yaml
import argparse
from src.env import Env
from src.multi_ue_env import MultiUeEnv

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
    if config["mode"] == "multiple_ue":
        env = MultiUeEnv(config)
    else:
        env = Env(config)
    
    # starting the simulator
    env.start()


if __name__ == "__main__":
    main()