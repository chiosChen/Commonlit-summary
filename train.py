import argparse

import wandb
import yaml 
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import numpy as np


import warnings

from src.data.data_utils import prepare_folds
from src.loss_fn.loss_function import get_score
from src.train_utils import train_loop
from src.utils.utils import init_wandb

warnings.filterwarnings("ignore")
sys.path.insert(0, "src")


os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-C", "--config", help="config filename")
    parser.add_argument("--device", type=int, default='0', required=False)   
    parser_args, _ = parser.parse_known_args(sys.argv)
    return parser.parse_args()


def get_logger(filename='./train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger()
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()

if __name__ == "__main__":
    cfg = parse_args()

    with open(cfg.config, 'r') as f:
        args = yaml.safe_load(f)

    args = SimpleNamespace(**args)
    args.device = cfg.device
    Path(args.checkpoints_path).mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/summaries_test.csv')
    prompt = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/prompts_train.csv')

    train_df = pd.merge(prompt, summary, on='prompt_id')

    TARGET = ['content', 'wording']
    seeds = [42]

    if args.wandb['use_wandb']:
        init_wandb()


    def get_result(oof_df):
        labels = oof_df[TARGET].values
        preds = oof_df[[f"pred_{c}" for c in TARGET]].values
        score, scores = get_score(labels, preds)
        LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')

    train_df = prepare_folds(train_df, args)
    if args.train:
        oof_df = pd.DataFrame()
        for fold in range(4):
            if fold in args.trn_fold:
                _oof_df = train_loop(train_df, fold, args)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)

    if args.wandb['use_wandb']:
        wandb.finish()

