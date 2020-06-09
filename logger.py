import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter

from utils import plot_tensor_to_numpy


class Logger(SummaryWriter):
    def __init__(self, logdir):
        if os.path.exists(logdir):
            raise RuntimeError(f'Logdir `{logdir} already exists. Remove it or specify new one.`')
        os.makedirs(logdir)
        super(Logger, self).__init__(logdir)
        
    def log(self, iteration, loss_stats):
        for key, value in loss_stats.items():
            if 'image' in key.split('/'):
                self.add_image(
                    key, plot_tensor_to_numpy(value),
                    iteration, dataformats='HWC'
                )
                continue
            self.add_scalar(key, value, iteration)
            
    def save_model_config(self, config):
        with open(f'{self.log_dir}/config.json', 'w') as f:
            json.dump(config, f)
    
    def save_checkpoint(self, iteration, model):
        filename = f'{self.log_dir}/checkpoint_{iteration}.pt'
        torch.save(model.state_dict(), filename)
