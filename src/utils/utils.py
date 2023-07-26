
import os
import random

import numpy as np
import pandas as pd


import warnings

from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
warnings.simplefilter("ignore")
import pickle


def make_sub(df):
    df = df.reset_index(drop=True)
    df[["discourse_id",'Ineffective','Adequate','Effective']].to_csv('submission.csv',index=False)

#==============================================================================================
def save_pickle(name,var):
    # print(f"Saving {name} ....")
    with open(name+'.pkl','wb') as fout:
        pickle.dump(var,fout)
    fout.close()
    
#==============================================================================================    
def load_pickle(name):
    # print(f"Loading {name} .....")
    with open(name,'rb') as fin:
        md = pickle.load(fin)
    fin.close()
    return md
#==============================================================================================
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # False


import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_loss(batch, model, return_outputs=True, device="cpu"):
    input_ids, attention_mask, targets = batch

    input_ids = input_ids.to(device).long()
    attention_mask = attention_mask.to(device).long()
    targets = targets.to(device).float()

    outputs = model(input_ids, attention_mask)
    outputs = outputs.sigmoid().squeeze(dim=-1)
    loss = F.mse_loss(outputs, targets, reduction="mean")

    return (loss, outputs) if return_outputs else loss

class AWP:
    def __init__(
            self,
            model,
            adv_param="weight",
            adv_lr=1,
            adv_eps=0.2,
            start_epoch=0,
            adv_step=1,
            device="cpu",
    ):
        self.model = model
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.device = device

    def attack_backward(self, batch, epoch=0):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        loss = 0
        self.save()
        for i in range(self.adv_step):
            self.attack_step()
            loss += calculate_loss(model=self.model, batch=batch, device=self.device, return_outputs=False)
            self.model.zero_grad()

        self.restore()

        return loss



    def attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self ,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


def init_wandb(args):

    import wandb

    try:

        wandb.login(key=args.wandb_api)
        anony = None
    except:
        anony = "must"
        print(
            'If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


    def class2dict(f):
        return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))

    run = wandb.init(project=args.wandb_project,
                     name=args.model,
                     config=class2dict(args),
                     group=args.model,
                     job_type="train",
                     anonymous=anony)


def get_optimizer_params(model, lr, weight_decay=0.0):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': lr, 'weight_decay': 0.0},
    ]
    return optimizer_parameters


def get_scheduler(cfg, optimizer, num_train_steps):
    if cfg.scheduler['name'] == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(cfg.scheduler['warmup'] * num_train_steps), num_training_steps=num_train_steps
        )
    elif cfg.scheduler['name'] == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(cfg.scheduler['warmup'] * num_train_steps), num_training_steps=num_train_steps,
            num_cycles=cfg.num_cycles

        )
    return scheduler

