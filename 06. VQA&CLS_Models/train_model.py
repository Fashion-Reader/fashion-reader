import os
from datetime import datetime, timezone, timedelta
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from models.get_model import get_model
from modules.dataset import get_datasets
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder
from modules.trainer import CustomTrainer
from modules.utils import load_yaml, save_yaml, get_logger, make_directory, seed_everything
import os

# CONFIG
PROJECT_DIR = "./"
ROOT_PROJECT_DIR = os.path.dirname(PROJECT_DIR)
TRAIN_CONFIG_PATH = os.path.join(PROJECT_DIR, 'config/train_config_v1.yaml')
config = load_yaml(TRAIN_CONFIG_PATH)

# DEBUG
TRAIN_TYPE = config['TRAIN']['type']

# SEED
RANDOM_SEED = config['SEED']['random_seed']


# MODEL
MODEL = config['MODEL']['model_str']

# TRAIN
BATCH_SIZE = config['TRAIN']['batch_size']
EPOCHS = config['TRAIN']['num_epochs']
LEARNING_RATE = config['TRAIN']['learning_rate']
EARLY_STOPPING_PATIENCE = config['TRAIN']['early_stopping_patience']
OPTIMIZER = config['TRAIN']['optimizer']
SCHEDULER = config['TRAIN']['scheduler']
MOMENTUM = config['TRAIN']['momentum']
WEIGHT_DECAY = config['TRAIN']['weight_decay']
LOSS_FN = config['TRAIN']['loss_function']
METRIC_FN = config['TRAIN']['metric_function']

# TRAIN SERIAL
KST = timezone(timedelta(hours=9))
TRAIN_TIMESTAMP = datetime.now(tz=KST).strftime("%Y%m%d%H%M%S")
TRAIN_SERIAL = f'{MODEL}_{TRAIN_TIMESTAMP}' if DEBUG is not True else 'DEBUG'

# PERFORMANCE RECORD
PERFORMANCE_RECORD_DIR = os.path.join(
    PROJECT_DIR, 'results', 'train', TRAIN_SERIAL)
PERFORMANCE_RECORD_COLUMN_NAME_LIST = config['PERFORMANCE_RECORD']['column_list']


if __name__ == '__main__':
    # Set random seed
    seed_everything(RANDOM_SEED)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set train result directory
    make_directory(PERFORMANCE_RECORD_DIR)

    # Set system logger
    system_logger = get_logger(name='train',
                               file_path=os.path.join(PERFORMANCE_RECORD_DIR, 'train_log.log'))

    # Load dataset & dataloader
    train_dataloader, validation_dataloader = get_datasets(config)

    # Load Model
    model = get_model(MODEL).to(device)

    # Set optimizer, scheduler, loss function, metric function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = None
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = accuracy_score

    # Set trainer
    trainer = CustomTrainer(model, device, loss_fn, metric_fn,
                            optimizer, scheduler, logger=system_logger)

    # Set earlystopper
    early_stopper = LossEarlyStopper(
        patience=EARLY_STOPPING_PATIENCE, verbose=True, logger=system_logger)

    # Set performance recorder
    key_column_value_list = [
        TRAIN_SERIAL,
        TRAIN_TIMESTAMP,
        MODEL,
        OPTIMIZER,
        LOSS_FN,
        METRIC_FN,
        EARLY_STOPPING_PATIENCE,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        RANDOM_SEED]

    performance_recorder = PerformanceRecorder(column_name_list=PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list,
                                               logger=system_logger,
                                               model=model,
                                               optimizer=optimizer,
                                               scheduler=None)

    # Save config yaml file
    save_yaml(os.path.join(PERFORMANCE_RECORD_DIR,
              'train_config_v1.yaml'), config)

    # Train
    for epoch in range(EPOCHS):
        trainer.train_epoch(train_dataloader, epoch_index=epoch, verbose=True)
        trainer.validate_epoch(validation_dataloader,
                               epoch_index=epoch, verbose=True)

        # Performance record - csv & save elapsed_time
        performance_recorder.add_row(epoch_index=epoch,
                                     train_loss=trainer.train_loss_mean,
                                     validation_loss=trainer.validation_loss_mean,
                                     train_score=trainer.train_score,
                                     validation_score=trainer.validation_score)

        # Performance record - plot
        performance_recorder.save_performance_plot(final_epoch=epoch)
        # early_stopping check
        early_stopper.check_early_stopping(loss=trainer.validation_loss_mean)

        if early_stopper.stop:
            break

        trainer.clear_history()

    # last model save
    performance_recorder.weight_path = os.path.join(
        PERFORMANCE_RECORD_DIR, 'last.pt')
    performance_recorder.save_weight()
