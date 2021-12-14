import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np


class CONFIG:
    seed = 318
    epochs = 5
    model_name = 'roberta-base'
    train_batch_size = 16
    valid_batch_size = 16
    learning_rate = 1e-4
    scheduler = "CosineAnnealingLR"
    min_learning_rate = 1e-6
    T_max = 500
    weight_decay = 1e-6
    n_classes = 1
    max_length = 150


    margin = .5
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
