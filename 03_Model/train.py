
import os
import copy
import torch
import pandas as pd

from adamp import AdamP
from torch.cuda.amp import GradScaler
from transformers import BertTokenizer
from datetime import datetime, timezone, timedelta

from models.get_model import get_model
from modules.training import *
from modules.utils import load_yaml, seed_everything


class CFG:
    # CONFIG
    PROJECT_DIR = "./"
    data_table_path = os.path.join(PROJECT_DIR, "data_table.csv")
    TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config_v1.yaml')
    config = load_yaml(TRAIN_CONFIG_PATH)

    # SEED
    seed = config['SEED']['random_seed']

    # MODEL
    model = config['MODEL']['model_str']

    # TRAIN
    epoch = config['TRAIN']['num_epochs']
    batch_size = config['TRAIN']['batch_size']
    lr = config['TRAIN']['learning_rate']
    early_stopping_patience = config['TRAIN']['early_stopping_patience']
    optimizer = config['TRAIN']['optimizer']
    scheduler = config['TRAIN']['scheduler']
    T_0 = config['TRAIN']['T_0']
    T_mult = config['TRAIN']['T_mult']
    min_lr = config['TRAIN']['min_lr']
    metric_fn = config['TRAIN']['metric_function']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TRAIN SERIAL
    KST = timezone(timedelta(hours=9))
    TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
    TRAIN_SERIAL = f'train_{model}_{TRAIN_TIMESTAMP}'

    # PERFORMANCE RECORD
    save_path = os.path.join(PROJECT_DIR, 'results', TRAIN_SERIAL)

if __name__ == '__main__':
    # Set random seed
    seed_everything(CFG.seed)
    
    # Set train result directory
    if not os.path.exists(CFG.save_path):
        os.makedirs(CFG.save_path)

    # Load data table
    df = pd.read_csv(CFG.data_table_path)

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load dataloader
    train_loader, val_loader = prepare_dataloader(df, tokenizer)

    # Load Model
    model = get_model(model_str=CFG.model).to(CFG.device)
    
    # Set optimizer
    if CFG.optimizer == 'AdamP':
        optimizer = AdamP(model.parameters(), lr=CFG.lr)
    else:
        raise NameError(f"There isn't {CFG.optimizer}")

    # Set scheduler
    if CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=CFG.T_mult, eta_min=CFG.min_lr ,last_epoch=-1)
    else:
        raise NameError(f"There isn't {CFG.scheduler}")
    
    # Set gradscaler
    scaler = GradScaler()

    # Check metric function
    if not CFG.metric_function in ['accuracy']:
        raise NameError(f"There isn't {CFG.metric_function}")

    best_accuracy = 0
    best_epoch = 0
    stop_count = 0
    for epoch in range(CFG.epoch):
        train_one_epoch(CFG, epoch, model, optimizer, train_loader, scaler, scheduler=scheduler)

        if CFG.metric_function == 'accuracy':
            with torch.no_grad():
                epoch_accuracy_dict = valid_one_epoch_accuracy(CFG, epoch, model, val_loader)
                epoch_accuracy = sum(epoch_accuracy_dict.values())/len(epoch_accuracy_dict)
                print(f"epoch{epoch}\taccuracy{epoch_accuracy:.4f}")

            if epoch_accuracy >= best_accuracy:
                stop_count = 0

                best_epoch = epoch
                best_accuracy = epoch_accuracy
                
                best_state_dict = copy.deepcopy(model.state_dict())
                if not os.path.exists(CFG.save_path):
                    os.mkdir(CFG.save_path)
                torch.save(model, os.path.join(CFG.save_path, f"Epoch{best_epoch}_Accuracy{best_accuracy:.4f}.pt"))
                print('The model is saved!')
            else:
                # early stopping
                stop_count += 1
                if stop_count > CFG.early_stopping_patience:
                    break
