# -*- coding: utf-8 -*-
"""

"""
import os
import toml
import argparse
from tqdm import tqdm
import copy
from data_loader import comm_to_embd, embd_to_comm, _get_data_size, sample_dataset_line
from models import get_init_net
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import wandb
from utils import *


def get_args_parser():
    # Training settings
    # ========= Usually default settings
    parser = argparse.ArgumentParser(description='Iterated ICL Toy')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')    
    parser.add_argument('--config_file', type=str, default=None,
                        help='the name of the toml configuration file')
    parser.add_argument('--seed', default=0, type=int)
    
    # ========= Model related
    parser.add_argument('--gpt_type',default='tiny',type=str,
                        help='what type of gpt2 we use, tiny, small, standard')

    # ========= Data related
    parser.add_argument('--train_data', default='tasks_train_simple',type=str,
                        help='txt file used during training')
    parser.add_argument('--eval_data', default='tasks_test_simple',type=str,
                        help='txt file used during training')
    
    # ========= IL setting
    parser.add_argument('--init_strategy', type=str, default='nil',
                    help='How to generate new student, nil or mile')
    parser.add_argument('--generations', type=int, default=1,
                        help='number of generations')
    
    # ========= Training
    parser.add_argument('--eval_interval', type=int, default=500,
                    help='The number of updates for interaction phase, train set has >16k samples') 
    parser.add_argument('--int_rounds', type=int, default=50000,
                    help='The number of updates for interaction phase, train set has >16k samples')
    parser.add_argument('--dis_rounds', type=int, default=1000,
                    help='The number of updates for imitation phase, train set has >16k samples')
    parser.add_argument('--dis_loss', type=str, default='ce_argmax',
                    help='how the teacher generate the samples, ce_argmax, ce_sample, noisy_ce_sample, mse')
    parser.add_argument('--s_ratio', type=int, default=0.6,
                    help='Split ratio of the training set, sup/unsup')    
    parser.add_argument('--dis_lr', default=1e-4, type=float,
                        help='the learning rate for training')
    parser.add_argument('--int_lr', default=1e-4, type=float,
                        help='the learning rate for training')  
    
    # ===== Wandb and saving results ====
    parser.add_argument('--WD_ID',default='joshuaren', type=str,
                        help='W&D ID, joshuaren or joshua_shawn')
    parser.add_argument('--save_model', default=False, type=eval, 
                        help='Whether save the model in the save-path') 
    parser.add_argument('--save_every_steps', default=10000, type=int,
                        help='gap between different savings')     
    parser.add_argument('--run_name',default='baseline_tiny',type=str)
    parser.add_argument('--proj_name',default='P5_iICL_toy', type=str)
    return parser

def get_acc(out_logi, y, split_idx):
    pred = out_logi.argmax(-1)[split_idx-1:]
    gt_y = y[split_idx-1:]
    part_acc = (pred[:-1]==gt_y[:-1]).sum()/gt_y.shape[0]     # partial mapping counts
    acc = part_acc.int()                            # See only complete correct predictions
    return part_acc, acc          


def interaction_phase(args, model, optimizer):
    # Student model train from training set directly
    #part_acc_avg = AverageMeter()
    model.train()
    for i in tqdm(range(args.int_rounds)):
        x, split_idx = sample_dataset_line(file_name=args.train_data, 
                                           sup_ratio=args.s_ratio, 
                                           sup_or_unsup="sup")
        x = x.to(args.device)
        y = torch.cat([x[1:], EOS_TENSOR])
        oht_x = F.one_hot(x, num_classes=24).float()
        optimizer.zero_grad()
        out_logi = model(oht_x)
        loss = nn.CrossEntropyLoss()(out_logi, y)
        part_acc, acc = get_acc(out_logi, y, split_idx)
        #part_acc_avg.update(part_acc)
        loss.backward()
        optimizer.step()
        if i % args.eval_interval==0:
            evaluate(args, model)
            print("Loss is %.4f, acc is %.4f"%(loss.data.item(),part_acc))
        wandb.log({'Inter_loss':loss.data.item()})
        wandb.log({'Train_pacc':part_acc})
        wandb.log({'Train_acc':acc})

def imitation_phase(args, student, teacher, optimizer):
    teacher.eval()
    student.train()
    for i in tqdm(range(args.dis_rounds)):
        x, split_idx = sample_dataset_line(file_name=args.train_data, 
                                           sup_ratio=args.s_ratio, 
                                           sup_or_unsup="unsup")
        x = x.to(args.device)
        oht_x = F.one_hot(x, num_classes=24).float()
        teach_logits = teacher(oht_x)
        stud_logits = student(oht_x)
        if args.dis_loss == "mse":
            loss = nn.MSELoss(reduction='mean')(stud_logits, teach_logits)
        else:
            if args.dis_loss == 'ce_argmax':
                sample_y = teach_logits.argmax(-1)          
            elif args.dis_loss == 'ce_sample':
                sampler = torch.distributions.categorical.Categorical(nn.Softmax(-1)(teach_logits))
                sample_y = sampler.sample().long()
            elif args.dis_loss == 'noisy_ce_sample':
                epsilon = torch.randn_like(teach_logits)
                sampler = torch.distributions.categorical.Categorical(nn.Softmax(-1)(teach_logits+epsilon))
                sample_y = sampler.sample().long()                
            teach_label = torch.cat([x[1:split_idx], sample_y[split_idx:], EOS_TENSOR])  
            loss = nn.CrossEntropyLoss()(stud_logits, teach_label)
        loss.backward()
        optimizer.step()
        wandb.log({'Distil_loss':loss.data.item()})

def evaluate(args, model):
    part_acc = AverageMeter()
    acc = AverageMeter()
    model.eval()
    N = _get_data_size(args.eval_data)
    for i in tqdm(range(N)):
        x, split_idx = sample_dataset_line(file_name=args.eval_data, idx=i)
        x = x.to(args.device)
        y = torch.cat([x[1:], EOS_TENSOR])
        oht_x = F.one_hot(x, num_classes=24).float()
        out_logi = model(oht_x)
        tmp_part, tmp_acc = get_acc(out_logi, y, split_idx)
        part_acc.update(tmp_part)
        acc.update(tmp_acc)
    #print("PACC is %.4f, ACC is %.4f"%(part_acc.avg, acc.avg))
    wandb.log({"Eval_pacc": part_acc.avg})
    wandb.log({"Eval_acc": acc.avg})
    model.train()
    return part_acc.avg, acc.avg
        

def main(args):
    if args.seed==0:
        args.seed = np.random.randint(1,10086)
    rnd_seed(args.seed)    
    # ========== Prepare save folder and wandb ==========
    wandb_init(proj_name=args.proj_name, run_name=args.run_name, config_args=args) 
    gens_valid_roc, gens_test_roc = [], []
    for gen in range(args.generations):
        # =========== Step0: new agent
        if args.init_strategy == 'nil':
            student = get_init_net(args)
        elif args.init_strategy == 'mile':
            if gen > 1:
                student = old_teacher
            else:
                student = get_init_net(args)
        else:
            student = get_init_net(args)        

        optimizer_int = optim.AdamW(student.parameters(), lr=args.dis_lr)
        optimizer_imi = optim.AdamW(student.parameters(), lr=args.int_lr)
        # =========== Step1: imitation, skip in first gen
        if gen > 0:
            imitation_phase(args, student, teacher, optimizer_imi)
            old_teacher = copy.deepcopy(teacher)     
        # =========== Step2: interaction
        interaction_phase(args, student, optimizer_int)
        teacher = copy.deepcopy(student)
    wandb.finish()


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.config_file is not None:
        config = toml.load(os.path.join("configs",args.config_file+".toml"))
        args = update_args(args, config)
    EOS_TENSOR = torch.tensor([23]).to(args.device)
    main(args)

    """ 
    model = build_model(args)
    optimizer_int = torch.optim.Adam(model.parameters(), lr=args.lr)
    for i in range(5):
        interaction_phase(args, model, optimizer_int)
        evaluate(args, model)
    #teacher = copy.deepcopy(model)
    #model = build_model(args)
    #optimizer_imi = torch.optim.Adam(model.parameters(), lr=args.lr)
    #imitation_phase(args, model, teacher, optimizer_imi)
    #part_acc, acc = evaluate(args, model)    

    x, split_idx = sample_dataset_line(file_name=args.eval_data, idx=1)
    x = x.to(args.device)
    oht_x = F.one_hot(x, num_classes=24).float()
    # ----- Feed all IN: + OUT: to the model
    out_logi = model(oht_x)
    pred = out_logi.argmax(-1)
    get_acc(out_logi, x, split_idx)
    
    # ----- Only feed IN: to the model
    query_x = x[:split_idx].to(args.device)
    oht_qx = F.one_hot(query_x, num_classes=24).float()
    out_qlogi = model(oht_qx)
    qpred = out_qlogi.argmax(-1)
    

    
    
    args = get_args_parser()
    args = args.parse_args()
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.config_file is not None:
        config = toml.load(os.path.join("configs",args.config_file+".toml"))
        args = update_args(args, config)
    main(args)
    
    """

