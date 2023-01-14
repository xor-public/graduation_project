from prepare_data import make_dataset
from client import Client
from server import Server
import argparse
import json
from loggings import logger

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--task",type=str,default="cifar_res18")
    args=parser.parse_args()
    # assert args.task in ["mnist","cifar10"]
    config=json.loads(open(f"config/{args.task}.json").read())
    logger.get_logger(config)
    logger.info(str(config))
    val_every_n_epochs=config["val_every_n_epochs"]
    train_data,val_data=make_dataset(config["dataset"])

    server=Server(config)
    clients=server.clients
    for client in clients:
        client.load_data(train_data)
    server.load_data(val_data)

    for epoch in range(config["epochs"]):
        epoch+=1
        logger.info(f"Epoch {epoch}")
        server.train_one_epoch()
        if epoch%val_every_n_epochs==0:
            server.validate()

if __name__ == "__main__":
    main()