import pandas as pd
from data_loader import ToxicTrainingDataset, ToxicValidationDataset
from config import set_seed, CONFIG
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler


def criterion(output1, output2, targets):
    return nn.MarginRankingLoss(margin=CONFIG.margin)(output1, output2, targets)


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    data_size = 0
    running_loss = 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    for step, data in bar:
        comment_ids = data['toxic_comment_ids'].to(device, dtype=torch.long)
        comment_mask = data['toxic_comment_mask'].to(device, dtype=torch.long)
        scores = data['score'].to(device, dtype=torch.long)

        batch_size = comment_ids.size(0)
        outputs = model(comment_ids, comment_mask)
        loss_function = nn.MSELoss()
        loss = loss_function(outputs, scores)
        # loss = criterion(more_toxic_outputs, less_toxic_outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()

        running_loss += (loss.item() * batch_size)
        data_size += batch_size
        epoch_loss = running_loss / data_size
        bar.set_postfix(EPOCH=epoch, TRAINING_LOSS=epoch_loss, LEARNING_RATE=optimizer.param_groups[0]['lr'])

    return epoch_loss


def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    data_size = 0
    running_loss = 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    predictions, real_results = np.ones(len(dataloader)), np.ones(len(dataloader))
    idx = 0
    for step, data in bar:
        with torch.no_grad():
            more_toxic_ids = data['more_toxic_ids'].to(device, dtype=torch.long)
            more_toxic_mask = data['more_toxic_mask'].to(device, dtype=torch.long)
            less_toxic_ids = data['less_toxic_ids'].to(device, dtype=torch.long)
            less_toxic_mask = data['less_toxic_mask'].to(device, dtype=torch.long)
            targets = data['target'].to(device, dtype=torch.long)

            batch_size = more_toxic_ids.size(0)
            more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
            less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)

            loss = criterion(more_toxic_outputs, less_toxic_outputs, targets)

            running_loss += (loss.item() * batch_size)
            data_size += batch_size
            epoch_loss = running_loss / data_size
            bar.set_postfix(EPOCH=epoch, VALIDATION_LOSS=epoch_loss)

            predictions[idx:idx + step] = targets
            idx += step
    print("Precision: ", (real_results == predictions).sum() / len(real_results))

    return epoch_loss


def prepare_dataloader():
    df_train = pd.read_csv('../task_a_distant.tsv')
    df_valid = pd.read_csv('../jigsaw-toxic-severity-rating/validation_data.csv')

    train_dataset = ToxicTrainingDataset(df_train, tokenizer=CONFIG.tokenizer, max_length=CONFIG.max_length)
    valid_dataset = ToxicValidationDataset(df_valid, tokenizer=CONFIG.tokenizer, max_length=CONFIG.max_length)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.train_batch_size, num_workers=2, shuffle=True,
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG.valid_batch_size, num_workers=2, shuffle=False,
                              pin_memory=True)
    return train_loader, valid_loader


def run_training(model, optimizer, scheduler, num_epochs):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    train_data_loader, valid_data_loader = prepare_dataloader()
    for epoch in range(1, num_epochs + 1):
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, dataloader=train_data_loader,
                                           device=CONFIG.device, epoch=epoch)
        valid_epoch_loss = valid_one_epoch(model, valid_data_loader, device=CONFIG.device, epoch=epoch)
        history['TRAIN_LOSS'].append(train_epoch_loss)
        history['VALID_LOSS'].append(valid_epoch_loss)

        if valid_epoch_loss < best_epoch_loss:
            print("The best loss was {}, Current Loss: {}".format(best_epoch_loss, valid_epoch_loss))
            best_epoch_loss = valid_epoch_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            PATH = f"model.bin"
            torch.save(model.state_dict(), PATH)
    print("best epoch loss ", best_epoch_loss)
    return model, history


def fetch_scheduler(optimizer):
    if CONFIG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG.T_max,
                                                   eta_min=CONFIG.min_learning_rate)
    elif CONFIG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG.T_0,
                                                             eta_min=CONFIG.min_learning_rate)
    elif CONFIG.scheduler == None:
        return None

    return scheduler
