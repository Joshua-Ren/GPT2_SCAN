# -*- coding: utf-8 -*-
"""
Modified from https://github.com/dtsip/in-context-learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm


def get_init_net(args):
    if args.gpt_type == 'tiny':
        n_embd, n_layer, n_head = 64, 3, 2
    elif args.gpt_type == 'small':
        n_embd, n_layer, n_head = 128, 6, 4
    elif args.gpt_type == 'standard':
        n_embd, n_layer, n_head = 256, 12, 8
    else:
        raise NotImplementedError        
    model = TransformerModel(
        n_dims=24,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    ).to(args.device)
    
    return model


class TransformerModel(nn.Module):
    def __init__(self, n_dims=24, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_dims)

    def forward(self, x):
        # x should be a line of message, like [IN: XXX OUT: XXX]
        tmp = self._read_in(x)
        tmp = self._backbone(inputs_embeds=tmp).last_hidden_state
        tmp_logi = self._read_out(tmp)
        return tmp_logi
    
    def generate(self, x, max_new_tokens=20):
        tmp = self._read_in(x)
        
        pass
        

if __name__ == "__main__":
    example_X = "IN: jump opposite right twice and turn opposite right thrice"
    example_Y = "OUT: I_TURN_RIGHT I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_TURN_RIGHT I_JUMP I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT I_TURN_RIGHT"
    import argparse
    def update_args(args, config):
        for k in config.keys():
            args.__dict__[k] = config[k]
        return args
    parser = argparse.ArgumentParser(description='test')
    config ={"family":"gpt2", "gpt_type": "small"}
    conf_args = update_args(parser,config)
    model = get_init_net(conf_args)
    n_dims = model.n_dims
    
























