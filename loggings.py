import logging
import os

class Logger:
    def __init__(self):
        self.accs=[0]
        self.backdool_accs=[0]
        self.num_poisons=[0]
        file_handler = logging.FileHandler('last_run.log')
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
    def add_log_file(self,name):
        if not os.path.exists('runs'):
            os.mkdir('runs')
        file_handler = logging.FileHandler('runs/'+name+'.log')
    def info(self,msg):
        self.logger.info(msg)
    def debug(self,msg):
        self.logger.debug(msg)

logger=Logger()