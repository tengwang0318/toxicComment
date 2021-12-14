from config import set_seed, CONFIG
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import torch

set_seed(CONFIG.seed)




class ToxicTrainingDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        super(ToxicTrainingDataset, self).__init__()
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.toxic_comment1 = df['txt1'].values
        self.toxic_comment2 = df['txt2'].values
        self.score1 = df['sc1'].values
        self.score2 = df['sc2'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.score1[index] >= self.score2[index]:
            more_toxic = self.toxic_comment1[index]
            less_toxic = self.toxic_comment2[index]
        else:
            more_toxic = self.toxic_comment2[index]
            less_toxic = self.toxic_comment1[index]
        inputs_more_toxic = self.tokenizer.encode_plus(
            more_toxic,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        inputs_less_toxic = self.tokenizer.encode_plus(
            less_toxic,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        target = 1
        more_toxic_ids, more_toxic_mask = inputs_more_toxic['input_ids'], inputs_more_toxic['attention_mask']
        less_toxic_ids, less_toxic_mask = inputs_less_toxic['input_ids'], inputs_less_toxic['attention_mask']
        return {
            'more_toxic_ids': torch.tensor(more_toxic_ids, dtype=torch.long),
            'more_toxic_mask': torch.tensor(more_toxic_mask, dtype=torch.long),
            'less_toxic_ids': torch.tensor(less_toxic_ids, dtype=torch.long),
            'less_toxic_mask': torch.tensor(less_toxic_mask, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }


class ToxicValidationDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        super(ToxicValidationDataset, self).__init__()
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.more_toxic = df['more_toxic'].values
        self.less_toxic = df['less_toxic'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        more_toxic = self.more_toxic[index]
        less_toxic = self.less_toxic[index]
        inputs_more_toxic = self.tokenizer.encode_plus(
            more_toxic,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        inputs_less_toxic = self.tokenizer.encode_plus(
            less_toxic,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        target = 1
        more_toxic_ids, more_toxic_mask = inputs_more_toxic['input_ids'], inputs_more_toxic['attention_mask']
        less_toxic_ids, less_toxic_mask = inputs_less_toxic['input_ids'], inputs_less_toxic['attention_mask']
        return {
            'more_toxic_ids': torch.tensor(more_toxic_ids, dtype=torch.long),
            'more_toxic_mask': torch.tensor(more_toxic_mask, dtype=torch.long),
            'less_toxic_ids': torch.tensor(less_toxic_ids, dtype=torch.long),
            'less_toxic_mask': torch.tensor(less_toxic_mask, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }
