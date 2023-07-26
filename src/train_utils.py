import re
import os
import math
import time
import json
import random
import numpy as np
import pandas as pd

from pathlib import Path

import torch
import wandb

from src.data.dataset import CustomDataset, Collatator
from src.utils.utils import get_optimizer_params, get_scheduler
from train import LOGGER


from torch.utils.data import DataLoader
from data.data_utils import batch_to_device, get_special_tokens, prepare_folds
from transformers import AutoTokenizer, AutoConfig, AutoModel, AdamW

from models.models import CommonLitModel
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup


from tqdm.auto import tqdm
import gc
import torch.utils.checkpoint
from loss_fn.loss_function import RMSELoss, MCRMSE, get_score


OUTPUT_DIR = './commonlit/train'

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# ------------------------------------------  ------------------------------------------- #
# ------------------------------------------  ------------------------------------------- #

class AutoSave:
  def __init__(self, top_k=3,metric_track="mae_val",mode="min", root=None):
    
    self.top_k = top_k
    self.logs = []
    self.metric = metric_track
    self.mode = -1 if mode=='min' else 1
    self.root = Path(root)
    assert self.root.exists()

    self.top_models = []
    self.top_metrics = []
    self.texte_log = []

  def log(self, model, metrics):
    metric = metrics[self.metric]
    rank = self.rank(self.mode*metric)

    self.top_metrics.insert(rank+1, self.mode*metric)
    if len(self.top_metrics) > self.top_k:
      self.top_metrics.pop(0)


    self.logs.append(metrics)
    self.save(model, rank, metrics)


  def save(self, model,rank, metrics):
    val_text = " "
    for k,v in metrics.items():
        if k in ["fold","epoch",'step','train_loss','val_loss']:
            if k in ["fold","epoch",'step']:
                val_text+=f"_{k}={v:.0f} "
            else:
                val_text+=f"_{k}={v:.4f} "

    name = val_text.strip()
    name = name+".pth"
    name = name.replace('=',"_")
    path = self.root.joinpath(name)

    old_model = None
    self.top_models.insert(rank+1, name)
    if len(self.top_models) > self.top_k:
      old_model = self.root.joinpath(self.top_models[0])
      self.top_models.pop(0)      

    torch.save(model.state_dict(), path.as_posix())

    if old_model is not None:
      old_model.unlink()


  def rank(self, val):
    r = -1
    for top_val in self.top_metrics:
      if val <= top_val:
        return r
      r += 1

    return r


# # ----------------- Opt/Sched --------------------- #
def get_optim_scheduler(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.optimizer["params"]['weight_decay'],
        "lr": args.optimizer["params"]['lr'],
                                    },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.optimizer["params"]['lr'],
        }]

    if args.optimizer['name'] == "optim.AdamW":
        optimizer = eval(args.optimizer['name'])(optimizer_grouped_parameters, lr=args.optimizer["params"]['lr'])
    else:
        optimizer = eval(args.optimizer['name'])(model.parameters(), **args.optimizer['params'])

    # if 'scheduler' in args:
    if args.scheduler['name'] == 'poly':

        params = args.scheduler['params']

        power = params['power']
        lr_end = params['lr_end']

        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))

        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end, power)

    elif args.scheduler['name'] in ['linear','cosine']:
        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))
        if args.scheduler['name']=="linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, training_steps)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)
            
    elif args.scheduler['name'] in ['optim.lr_scheduler.OneCycleLR']:
        max_lr = args.optimizer['params']['lr']
        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))
        scheduler = eval(args.scheduler['name'])(optimizer,max_lr=max_lr,
                                                 epochs=args.trainer['epochs'],
                                                 steps_per_epoch=training_steps,
                                                 pct_start = args.scheduler['warmup']
                                                 )

    return optimizer, scheduler


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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device, args):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=args.apex)
    losses = AverageMeter()
    global_step = 0

    for step, batch in enumerate(tqdm(train_loader)):
        batch = batch_to_device(batch, device)
        labels = batch.pop('label')
        batch_size = labels.size(0)

        with torch.cuda.amp.autocast(enabled=args.apex):
            pred = model(batch)
            loss = criterion(pred, labels)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        losses.update(loss.item(), batch_size)

        if args.apex:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if args.max_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=args.max_norm
            )

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if args.batch_scheduler:
                scheduler.step()


        if args.wandb:
            wandb.log({f"[fold{fold}] loss": losses.val,
                       f"[fold{fold}] lr": scheduler.get_lr()[0]})

    return losses.avg


def valid_fn(valid_loader, model, criterion, device, args):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, batch in enumerate(tqdm(valid_loader)):
        batch = batch_to_device(batch, device)
        labels = batch.pop('label')
        batch_size = labels.size(0)

        with torch.no_grad():
            pred = model(batch)

            loss = criterion(pred, labels)

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        losses.update(loss.item(), batch_size)

        preds.append(pred.squeeze(0).to('cpu').numpy())

        end = time.time()
        if step % args.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step + 1) / len(valid_loader))))

    predictions = np.concatenate(preds)

    return losses.avg, predictions


def train_loop(folds, fold, args):
    LOGGER.info(f"========== fold: {fold} training ==========")

    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[args.target_cols].values

    special_tokens = get_special_tokens()
    tokenizer = AutoTokenizer.from_pretrained(args.model['model_name'], use_fast=True,
                                              additional_special_tokens=list(special_tokens.values()))
    tokenizer.save_pretrained('./tokenizers')

    train_dataset = CustomDataset(train_folds, tokenizer, args)
    valid_dataset = CustomDataset(valid_folds, tokenizer, args)

    train_collator = Collatator(args)
    valid_collator = Collatator(args)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.train_loader['batch_size'],
                              shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              collate_fn=train_collator
                              )
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.val_loader['batch_size'],
                              shuffle=args.val_loader['shuffle'],
                              num_workers=args.val_loader['num_workers'], pin_memory=True, drop_last=False,
                              collate_fn=valid_collator
                              )

    config = AutoConfig.from_pretrained(args.model_name, output_hidden_states=True)
    config.update(

        {
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            'max_position_embeddings': 2048
        }
    )
    backbone = AutoModel.from_pretrained(args.model_name, config=config)

    backbone.resize_token_embeddings(len(tokenizer))

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    model = CommonLitModel(backbone, pooling_params= args.model['pooling_params'], use_gradient_checkpointing=args.trainer['use_gradient_checkpointing'])

    torch.save(model.config, OUTPUT_DIR + 'config.pth')
    model.to(device)

    optimizer_parameters = get_optimizer_params(model,
                                                lr = args.optimizer['params']['lr'],
                                                weight_decay = args.optimizer['params']['weight_decay'])

    optimizer = AdamW(optimizer_parameters, lr=args.optimizer['params']['lr'], eps=args.optimizer['params']['eps'], betas=args.optimizer['params']['betas'])

    num_train_steps = int(len(train_folds) / args.train_loader['batch_size'] * args.trainer['epochs'])

    scheduler = get_scheduler(args, optimizer, num_train_steps)

    criterion = RMSELoss(reduction="mean")

    best_score = np.inf

    model.zero_grad()

    for epoch in range(args.trainer['epochs']):

        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)

        # scoring
        score, scores = get_score(valid_labels, predictions)

        if args.wandb['use_wandb']:
            wandb.log({f"[fold{fold}] epoch": epoch + 1,
                       f"[fold{fold}] avg_train_loss": avg_loss,
                       f"[fold{fold}] avg_val_loss": avg_val_loss,
                       f"[fold{fold}] score": score})

        if best_score > score:
            best_score = score
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')


    predictions = torch.load(OUTPUT_DIR + f"{args.model_name.replace('/', '-')}_fold{fold}_best.pth",
                             map_location=torch.device('cpu'))['predictions']

    valid_folds[[f"pred_{c}" for c in args.target]] = predictions

    torch.cuda.empty_cache()

    gc.collect()

    return valid_folds


    

