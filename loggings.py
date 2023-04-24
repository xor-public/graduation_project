import logging
import os
import yaml

class Logger:
    def __init__(self):
        self.accs=[0]
        self.backdool_accs=[0]
        self.num_poisons=[0]
        self.config=None
        self.set_logger()
        self.clean_tmp()
    def set_logger(self):
        file_handler = logging.FileHandler('last_run.log',mode='w')
        console_handler = logging.StreamHandler()
        file_handler.setLevel('DEBUG')
        console_handler.setLevel('INFO')
        fmt = '%(asctime)s \n%(message)s\n'
        formatter = logging.Formatter(fmt)
        file_handler.setFormatter(formatter)
        # console_handler.setFormatter(formatter)
        logger = logging.getLogger('logger')
        logger.setLevel('DEBUG')     
        logger.addHandler(file_handler)    
        logger.addHandler(console_handler)
        self.logger=logger
    def clean_tmp(self):
        if os.path.exists("./tmp"):
            os.system("rm -rf ./tmp/*> /dev/null 2>&1")
        else:
            os.mkdir("./tmp")
    def set_args(self,args):
        self.args=args
        config=yaml.safe_load(open(f"config/{args.task}.yaml",'r').read())
        self.set_config(config)
        self.add_log_file(f"{args.task}_{args.attack_method}_{args.defend_method}.log")
        if args.attack_method:
            self.config["fl"]=self.config["poisoned_fl"]
            self.info(f"fl: {config['fl']}")
            self.info(f"Attack method: {args.attack_method}")
            self.info(f"{args.attack_method}: {config[args.attack_method]}")
        else:
            self.info(f"fl: {config['fl']}")
    def set_config(self,config):
        self.config=config
    def add_log_file(self,name):
        if not os.path.exists('runs'):
            os.mkdir('runs')
        file_handler = logging.FileHandler('runs/'+name+'.log',mode='w')
        fmt = '%(asctime)s \n%(message)s\n'
        formatter = logging.Formatter(fmt)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    def info(self,msg):
        self.logger.info(msg)
    def debug(self,msg):
        self.logger.debug(msg)

logger=Logger()