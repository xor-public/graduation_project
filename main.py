from prepare_data import make_dataset
from client import Client
from server import Server
import argparse
import json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--task",type=str,default="mnist_cnn")
    args=parser.parse_args()
    # assert args.task in ["mnist","cifar10"]
    config=json.loads(open(f"config/{args.task}.json").read())
    train_data,val_data=make_dataset(config["dataset"])

    server=Server(config)
    clients=server.clients
    for client in clients:
        client.load_data(train_data)
    server.load_data(val_data)

    for epoch in range(config["epochs"]):
        print(f"Epoch {epoch}")
        server.train_one_epoch()
        server.validate()

if __name__ == "__main__":
    main()