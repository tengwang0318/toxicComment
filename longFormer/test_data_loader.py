from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import pandas as pd
from test_config import TestConfig


class TestDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text = df['text'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long)
        }


def create_test_data_loader():
    df = pd.read_csv('../jigsaw-toxic-severity-rating/comments_to_score.csv')
    test_dataset = TestDataset(df, TestConfig.tokenizer, max_length=TestConfig.max_length)
    test_loader = DataLoader(test_dataset, batch_size=TestConfig.test_batch_size,
                             num_workers=2, shuffle=False, pin_memory=True)
    return test_loader


