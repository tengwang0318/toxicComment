from config import set_seed, CONFIG
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
import torch

set_seed(CONFIG.seed)


def create_Folds():
    df = pd.read_csv('../jigsaw-toxic-severity-rating/myData.csv')
    skf = StratifiedKFold(n_splits=CONFIG.n_fold, shuffle=True, random_state=CONFIG.seed)

    for fold, (_, val) in enumerate(skf.split(X=df, y=df.worker)):
        df.loc[val, "Kfold"] = int(fold)

    df['Kfold'] = df['Kfold'].astype(int)
    return df


class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        super(ToxicDataset, self).__init__()
        self.df = df
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.text1 = df['text1'].values
        self.text2 = df['text2'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text1 = self.text1[index]
        text2 = self.text2[index]
        inputs_text1 = self.tokenizer.encode_plus(
            text1,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )
        inputs_text2 = self.tokenizer.encode_plus(
            text2,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length'
        )

        label = self.labels[index]
        text1_ids, text1_mask = inputs_text1['input_ids'], inputs_text1['attention_mask']
        text2_ids, text2_mask = inputs_text2['input_ids'], inputs_text2['attention_mask']

        return {
            'text1_ids': torch.tensor(text1_ids, dtype=torch.long),
            'text1_mask': torch.tensor(text1_mask, dtype=torch.long),
            'text2_ids': torch.tensor(text2_ids, dtype=torch.long),
            'text2_mask': torch.tensor(text2_mask, dtype=torch.long),
            'target': torch.tensor(label, dtype=torch.long)
        }
