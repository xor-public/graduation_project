from prepare_data import make_dataset
from client import Client
from server import Server
from attacks.attack import Attacker
import argparse
import json
import torch
import numpy as np
import random
from loggings import logger

def main():
    set_fixed_seed(42)

    args=parse_args()

    config=json.loads(open(f"config/{args.task}.json").read())

    logger.add_log_file(f"{args.task}_{args.attack_method}_{args.defend_method}.log")
    for key in config.keys():
        logger.info(f"{key}: {config[key]}")

    server=Server(config)
    clients=server.clients
    if args.resume:
        server.model.load_state_dict(torch.load(args.resume))
    
    train_data,val_data,data_split=make_dataset(config["dataset"],attack_method=args.attack_method,split="non_iid",num_clients=config["K"])

    for client in clients:
        client.load_data(train_data,data_split)
    server.load_data(val_data)

    if args.attack_method:
        attacker=Attacker(args.attack_method)
        attacker.attack(clients)

    for epoch in range(config["epochs"]):
        epoch+=1
        logger.info(f"Epoch {epoch}")
        server.train_one_epoch()
        val_every_n_epochs=1
        if epoch%val_every_n_epochs==0:
            logger.accs.append(server.validate()[1])
            if args.attack_method:
                if attacker.backdoor_task:
                    logger.backdool_accs.append(server.model.validate(attacker.method.backdoor_test_loader,mode='backdoor')[1])

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
    parser.add_argument("-t","--task",type=str,default="mnist_mlp")
    parser.add_argument("-r","--resume",type=str,default=None)
    parser.add_argument("-a","--attack_method",type=str,default=None)
    parser.add_argument("-d","--defend_method",type=str,default=None)
    parser.add_argument("--iid",action="store_true",default=False)
    parser.add_argument("--low_vram",action="store_true",default=False)

    args=parser.parse_args()
    return args

if __name__ == "__main__":
    main()