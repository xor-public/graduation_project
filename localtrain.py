from models.model_loader import ModelLoader
from prepare_data import make_dataset
import argparse
import torch
import json
import yaml
import random
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-t","--task",type=str,default="cifar_cnn")
    parser.add_argument("-r","--resume",type=str,default=None)
    args=parser.parse_args()
    # config=json.loads(open(f"config/{args.task}.json").read())
    config=yaml.safe_load(open(f"config/{args.task}.yaml").read())
    config["local"]["model"]=config["model"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=ModelLoader(config).load_model().to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    # print(sum([p.numel() for p in model.parameters()]))
    train_data,val_data,_=make_dataset(config["dataset"])
    config=config["local"]
    if args.task!='word':
        train_loader = torch.utils.data.DataLoader(train_data,batch_size=config["batch_size"],shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data,batch_size=config["val_batch_size"])

    else:
        random.shuffle(train_data)
        val_loader = torch.utils.data.DataLoader(val_data,batch_size=64)
        val_loader = []
        for i in range(0,len(val_data),64):
            val_loader.append(val_data[i:i+65])
    # if 'cifar' in args.task or 'tinyimagenet' in args.task:
    if args.task=='cifar_cnn':
        max_epoch=config["epochs"]
        optimizer = torch.optim.SGD(model.parameters(),**config["optimizer_args"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
    elif args.task=='cifar_res18' or args.task=='tiny_imagenet':
        max_epoch=config["epochs"]
        optimizer = torch.optim.SGD(model.parameters(),**config["optimizer_args"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
    elif args.task=='mnist_mlp' or args.task=='mnist_cnn':
        max_epoch=config["epochs"]
        optimizer = torch.optim.SGD(model.parameters(),**config["optimizer_args"])
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0.01,max_epoch)
    elif args.task=='word':
        max_epoch=1
        optimizer = torch.optim.SGD(model.parameters(),**config["optimizer_args"])#,momentum=0.9,weight_decay=5e-4)
        # optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 0.01,max_epoch)
    best_acc=0
    for epoch in range(max_epoch):
        print("Epoch: ",epoch)
        if args.task=='word':
            print("val every 100 steps")
            steps=0
            for data_count in range(800):
                print(data_count*100,"steps")
                if steps%100==0:
                    _,acc=model.validate(val_loader)
                    if acc>best_acc:
                        best_acc=acc
                        torch.save(model.state_dict(), f'checkpoints/{config["model"]}.pt')
                        print('save',acc)
                for data in tqdm(train_data[data_count*100:(data_count+1)*100]):
                    train_loader = torch.utils.data.DataLoader(data,batch_size=64)
                    bs=batch_size=64
                    train_loader = []
                    for i in range(0,len(data),bs):
                        train_loader.append(data[i:i+bs+1])
                    for i in range(2):
                        model.train_one_epoch(train_loader,optimizer,progress_bar=False)
                    steps+=1
        else:
            model.train_one_epoch(train_loader,optimizer,progress_bar=True)
        _,acc=model.validate(val_loader)
        if acc>best_acc:
            best_acc=acc
            torch.save(model.state_dict(), f'checkpoints/{config["model"]}.pt')
            print('save',acc)
        scheduler.step()