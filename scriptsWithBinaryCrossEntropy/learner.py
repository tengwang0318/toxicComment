from config import set_seed, CONFIG
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import numpy as np
from collections import defaultdict
from data_loader import create_Folds, ToxicDataset
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import precision_score


def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    data_size = 0
    running_loss = 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    loss_function = nn.BCELoss()
    for step, data in bar:
        text1_ids = data['text1_ids'].to(device, dtype=torch.long)
        text1_mask = data['text1_mask'].to(device, dtype=torch.long)
        text2_ids = data['text2_ids'].to(device, dtype=torch.long)
        text2_mask = data['text2_mask'].to(device, dtype=torch.long)
        targets = data['target'].to(device, dtype=torch.float)

        batch_size = text1_ids.size(0)
        predicts = model(text1_ids, text1_mask, text2_ids, text2_mask)
        predicts = predicts.squeeze()
        loss = loss_function(predicts, targets)

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


def valid_one_epoch(model, dataloader, device, epoch, predictions, real_results):
    model.eval()
    data_size = 0
    running_loss = 0
    # predictions, real_results = np.ones((len(dataloader), 1)), np.ones((len(dataloader), 1))
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    loss_function = nn.BCELoss()
    idx = 0
    for step, data in bar:
        with torch.no_grad():
            text1_ids = data['text1_ids'].to(device, dtype=torch.long)
            text1_mask = data['text1_mask'].to(device, dtype=torch.long)
            text2_ids = data['text2_ids'].to(device, dtype=torch.long)
            text2_mask = data['text2_mask'].to(device, dtype=torch.long)
            targets = data['target'].to(device, dtype=torch.float)

            batch_size = text1_ids.size(0)
            outputs = model(text1_ids, text1_mask, text2_ids, text2_mask)
            outputs = outputs.squeeze()

            loss = loss_function(outputs, targets)

            running_loss += (loss.item() * batch_size)
            data_size += batch_size
            epoch_loss = running_loss / data_size
            bar.set_postfix(EPOCH=epoch, VALIDATION_LOSS=epoch_loss)

            predictions[idx:idx + batch_size] = outputs.cpu().squeeze() > 0.5
            real_results[idx:idx + batch_size] = targets.cpu().squeeze()

            idx += batch_size

    print("Precision: ", precision_score(real_results, predictions))

    return epoch_loss


def prepare_dataloader(fold):
    df = create_Folds()
    df_train = df[df.Kfold != fold].reset_index(drop=True)
    df_valid = df[df.Kfold == fold].reset_index(drop=True)
    # df_train = df_train[:int(len(df_train)*0.01)]
    # df_valid = df_valid[:int(len(df_valid)*0.01)]
    train_dataset = ToxicDataset(df_train, tokenizer=CONFIG.tokenizer, max_length=CONFIG.max_length)
    valid_dataset = ToxicDataset(df_valid, tokenizer=CONFIG.tokenizer, max_length=CONFIG.max_length)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.train_batch_size, num_workers=2, shuffle=True,
                              pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG.valid_batch_size, num_workers=2, shuffle=False,
                              pin_memory=True)

    predictions, real_results = np.ones((len(df_valid))), np.ones((len(df_valid)))

    return train_loader, valid_loader, predictions, real_results


def run_training(model, optimizer, scheduler, num_epochs, fold):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)
    train_data_loader, valid_data_loader, predictions, real_results = prepare_dataloader(fold)
    for epoch in range(1, num_epochs + 1):
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, dataloader=train_data_loader,
                                           device=CONFIG.device, epoch=epoch)
        valid_epoch_loss = valid_one_epoch(model, valid_data_loader, device=CONFIG.device, epoch=epoch,
                                           predictions=predictions, real_results=real_results)
        history['TRAIN_LOSS'].append(train_epoch_loss)
        history['VALID_LOSS'].append(valid_epoch_loss)

        if valid_epoch_loss < best_epoch_loss:
            print("The best loss was {}, Current Loss: {}".format(best_epoch_loss, valid_epoch_loss))
            best_epoch_loss = valid_epoch_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            PATH = f"FOLD-{fold}.bin"
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

# def custom_scheduler(optimizer, steps_per_epoch, lr_min, lr_max):
#     def lr_lambda(current_step):
#         current_step = current_step % steps_per_epoch
#         if current_step < steps_per_epoch / 2:
#             y = ((lr_max - lr_min) / (steps_per_epoch / 2)) * current_step + lr_min
#         else:
#             y = (-1.0 * (lr_max - lr_min) / (steps_per_epoch / 2)) * current_step + lr_max + (lr_max - lr_min)
#         return y
#
#     return LambdaLR(optimizer, lr_lambda, last_epoch=-1)
#
#
# def fetch_scheduler(optimizer):
#     sch = custom_scheduler(
#         optimizer,
#         self.steps_per_epoch,
#         1e-6,
#         1.5e-5,
#     )
#     return sch
