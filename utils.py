# -*- coding: utf-8 -*-
"""
"""
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
import numpy as np
import torch.backends.cudnn as cudnn

def update_args(args, config):
    for k in config.keys():
        args.__dict__[k] = config[k]
    return args

# ============== Wandb =======================
def wandb_init(proj_name='test', run_name=None, config_args=None, entity="joshuaren"):
    if config_args is not None:
        entity = config_args.WD_ID
    wandb.init(
        project=proj_name,
        config={}, entity=entity,reinit=True)
    if config_args is not None:
        wandb.config.update(config_args)
    if run_name is not None:
        wandb.run.name=run_name
        return run_name
    else:
        return wandb.run.name

def rnd_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count