import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np


class CONFIG:
    seed = 318
    epochs = 3
    model_name = 'allenai/longformer-base-4096'
    train_batch_size = 4
    valid_batch_size = 4
    learning_rate = 1e-5
    scheduler = "CosineAnnealingLR"
    min_learning_rate = 1e-7
    T_max = 500
    weight_decay = 1e-7
    n_fold = 5
    n_classes = 1
    max_length = 4096
    margin = .1
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
