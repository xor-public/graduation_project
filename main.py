from server import Server
from attacks.attack import Attacker
import argparse
import torch
import numpy as np
from utils.text_load import *
import random
from loggings import logger

def main():
    set_fixed_seed(42)

    args=parse_args()

    logger.set_args(args)

    server=Server()

    if args.attack_method:
        attacker=Attacker()
        attacker.attack(server)
        attacker.run()
    else:
        server.run()
    # if args.attack_method:
    #     if not config[args.attack_method]["single_shot"]:
    #         attacker=Attacker(config,args.attack_method)
    #         attacker.attack(clients)
    #     else:
    #         attack_epochs=config[args.attack_method]["attack_epochs"]

def set_fixed_seed(seed):
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("-t","--task",type=str,required=True)
    parser.add_argument("-r","--resume",type=str,default=None)
    parser.add_argument("-a","--attack_method",type=str,default=None)
    parser.add_argument("-d","--defend_method",type=str,default=None)
    parser.add_argument("--iid",action="store_true",default=False)
    parser.add_argument("--lowvram",action="store_true",default=False)

    args=parser.parse_args()
    return args

if __name__ == "__main__":
    main()