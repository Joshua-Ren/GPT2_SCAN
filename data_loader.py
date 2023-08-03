# -*- coding: utf-8 -*-
"""
    Name                      Length
    tasks_train_length        16990
    tasks_train_simple        16728
    tasks_test_simple         4182

See the length of each txt file

"""
import torch
import os
import numpy as np

def _get_data_size(file_name="tasks_test_simple"):
    data_path = os.path.join("dataset",file_name+".txt")
    with open(data_path, "r") as f:
        tmp = f.read()
    all_lines = tmp.split("\n")[:-1]
    return len(all_lines)

def _get_dicts():
    DATA_PATH = os.path.join("dataset","tasks_train_length.txt")
    # ----------- Scan the whole training set, build the embedding dictionary
    with open(DATA_PATH, "r") as f:
        tmp = f.read()
    
    comm_embd = {"IN:": 0, "OUT:": 1, "\n": 2,"EOS": 23}
    embd_comm = {0:"IN:", 1:"OUT:",2:"\n", 23:"EOS"}
    
    emb_value = 3
    
    for line in tmp.split("\n"):
        tmp_words = line.split(" ")
        for s in tmp_words:
            if s not in comm_embd:
                comm_embd[s] = emb_value
                embd_comm[emb_value] = s
                emb_value += 1
    return comm_embd, embd_comm

comm_embd, embd_comm = _get_dicts()

def comm_to_embd(line):
    tmp_embd = []
    cnt = 0
    split_idx = 0
    for s in line.split(" "):
        tmp_embd.append(comm_embd[s])
        cnt += 1
        if s == "OUT:":
            split_idx = cnt
    return torch.tensor(tmp_embd), split_idx

def embd_to_comm(embds):
    tmp_line = ""
    for i in range(embds.shape[0]):
        e = embds[i].item()
        s = embd_comm[e]
        tmp_line += s
        tmp_line += " "
    return tmp_line[:-1]

# ---------------- The training set
def sample_dataset_line(file_name="tasks_train_simple",
                        sup_ratio=0.6, 
                        sup_or_unsup="sup",
                        idx=None):
    
    data_path = os.path.join("dataset",file_name+".txt")
    # ----------- Scan the whole training set, build the embedding dictionary
    with open(data_path, "r") as f_train:
        tmp_train = f_train.read()
    all_lines = tmp_train.split("\n")[:-1]
    N_Train = int(sup_ratio*len(all_lines))   # [:NT] is train [NT:] is unsup_train
    if idx is None:
        if sup_or_unsup == "sup":
            idx = torch.randint(low=0, high=N_Train, size=(1,)).item()
        else:
            idx = torch.randint(low=N_Train+1, high=len(all_lines), size=(1,)).item()
    idx = np.mod(idx,len(all_lines))
    x_line = all_lines[idx]
    x, split_idx = comm_to_embd(x_line)
    return x, split_idx

if __name__ == "__main__":

    """
  import argparse
    from models import build_model
    def update_args(args, config):
        for k in config.keys():
            args.__dict__[k] = config[k]
        return args
    parser = argparse.ArgumentParser(description='test')
    config ={"gpt_type": "small","device":"cuda"}
    conf_args = update_args(parser,config)
    model = build_model(conf_args)
    teacher = copy.deepcopy(model)
    
    
    CE_LOSS = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    for i in range(10):
        x, split_idx = comm_to_embd(example_line)
        x = x.to(conf_args.device)
        y = x
        oht_x = F.one_hot(x,num_classes=24).float()
        out_logi = model(oht_x)
        # -------- Interaction
        loss = CE_LOSS(out_logi, y)
        # -------- Imitation
        #teach_logits = teacher(oht_x)
        #sampler = torch.distributions.categorical.Categorical(nn.Softmax(-1)(teach_logits))
        #sample_y = sampler.sample().long()
        #loss = CE_LOSS(out_logi[split_idx:], sample_y[split_idx:])
        # -------- Evaluation
    
        pred = out_logi.argmax(-1)[split_idx:]
        gt_y = y[split_idx:]
        part_acc = (pred==gt_y).sum()/gt_y.shape[0]     # partial mapping counts
        acc = part_acc.int()                            # See only complete correct predictions
        
        loss.backward()
        optimizer.step()
        print(loss.item())
        print(embd_to_comm(out_logi.argmax(-1).cpu()))
    """


















