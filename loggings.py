import logging
import os

class Logger:
    def __init__(self):
        pass
    def get_logger(self,config):
        task_name=config["model"]
        if not os.path.exists('runs'):
            os.mkdir('runs')
        for i in range(999):
            if not os.path.exists(f'./runs/{task_name}_{i}.log'):
                file_handler = logging.FileHandler(f'./runs/{task_name}_{i}.log')
                console_handler = logging.StreamHandler()
                file_handler.setLevel('DEBUG')
                console_handler.setLevel('INFO')
                fmt = '%(asctime)s \n%(message)s\n'
                formatter = logging.Formatter(fmt)
                file_handler.setFormatter(formatter)
                console_handler.setFormatter(formatter)
                logger = logging.getLogger('logger')
                logger.setLevel('DEBUG')     
                logger.addHandler(file_handler)    
                logger.addHandler(console_handler)
                break
        self.logger=logger
    def info(self,msg):
        self.logger.info(msg)
    def debug(self,msg):
        self.logger.debug(msg)

logger=Logger()