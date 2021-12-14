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
        self.toxic_comments = df['text'].values
        self.scores = df['average'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            self.toxic_comments[index],
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        score = self.scores[index]
        toxic_comment_ids, toxic_comment_mask = inputs['input_ids'], inputs['attention_mask']
        return {
            'toxic_comment_ids': torch.tensor(toxic_comment_ids, dtype=torch.long),
            'toxic_comment_mask': torch.tensor(toxic_comment_mask, dtype=torch.long),
            'score': torch.tensor(score, dtype=torch.long)
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
