# About this project
This project is a implementation of Federated Learning and Backdoor Attack, and some not such useful defend methods. 
# To run
## Local train
```bash
python localtrain.py --task [taskname] [--resume [modelname]]
```
this command will train a model on local dataset \
implemented tasks are **MNIST, CIFAR10, REDDIT** \
for **MNIST** the model is a simple MLP and a CNN \
for **CIFAR10** the model is a simple CNN and ResNet18 \
for **REDDIT** the model is a 2-layer single-direction LSTM\
taskname in [ **mnist_mlp, mnist_cnn, cifar_cnn, cifar_res18, word** ] is supported
full config is in ./config/*.yaml\
if you want to run Reddit dataset, you need to download [corpus_80000.pt.tar](https://drive.google.com/file/d/1qTfiZP4g2ZPS5zlxU51G-GDCGGr23nvt/view?usp=sharing) and [50k_word_dictionary.pt](https://drive.google.com/file/d/1gnS5CO5fGXKAGfHSzV3h-2TsjZXQXe39/view?usp=sharing) and put them in ./data/reddit/, other datasets can be downloaded automatically
## Federated train
```bash
python main.py --task [taskname] [--iid] [--lowvram] [--resume [modelname]]
```
this command will train a model on federated learning\
I implemented FedAvg on above tasks\
add flag [--iid] will train a model on IID dataset, by default it is non-IID\
add flag [--lowvram] all the clients' training will use the same model, every model will be saved in ./tmp after training, before AGG we load them, this will save a lot of GPU memory\

## Backdoor attack
```bash
python main.py --task [taskname] --attack_method [attack_method] [--iid] [--lowvram] [--resume [modelname]]
```
this command will train a model on federated learning with backdoor attack\
implemented attack methods are **Untargeted Poisoning, Constrain and Scale, DBA, Edgecase, PGD**
## Defend
```bash
python main.py --task [taskname] --attack_method [attack_method] --defend_method [defend_method] [--iid] [--lowvram] [--resume [modelname]]
```
this command will train a model on federated learning with backdoor attack and defend method
implemented defend methods are **multi-krum, median**, mymethod

<!-- ## Other words
I promise great implement on Federated Learning and Backdoor Attack. Defend is not an interesting thing, which cost me a lot of time and I don't feel it useful.\
I agree with the existence of backdoor attack, but is it really a problem?\
Attacker should take control of a lot of clients, I don't think it is such possible in a big system.\
if the attacker did it, the only word I can say is "the FL System is so weak that it can be attacked by a single person" -->