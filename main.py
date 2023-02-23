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
    parser=argparse.ArgumentParser()
    parser.add_argument("-t","--task",type=str,default="mnist_mlp")
    parser.add_argument("-a","--attack",action="store_true")
    parser.add_argument("-m","--attack_method",type=str,default="none")
    args=parser.parse_args()

    # assert args.task in ["mnist","cifar10"]
    config=json.loads(open(f"config/{args.task}.json").read())
    logger.get_logger(config)
    for key in config.keys():
        logger.info(f"{key}: {config[key]}")
    try:
        val_every_n_epochs=config["val_every_n_epochs"]
    except:
        val_every_n_epochs=1

    server=Server(config)
    clients=server.clients
    
    if args.attack:
        train_data,val_data=make_dataset(config["dataset"],args.attack_method)
    else:
        train_data,val_data=make_dataset(config["dataset"])

    for client in clients:
        client.load_data(train_data)
    server.load_data(val_data)

    if args.attack:
        attacker=Attacker(args.attack_method)
        attacker.attack(clients)

    for epoch in range(config["epochs"]):
        epoch+=1
        logger.info(f"Epoch {epoch}")
        server.train_one_epoch()
        if epoch%val_every_n_epochs==0:
            server.validate()
            if args.attack:
                if attacker.backdoor_task:
                    attacker.method.accs.append(server.model.validate(attacker.method.backdoor_test_loader)[1])

def set_fixed_seed(seed):
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()