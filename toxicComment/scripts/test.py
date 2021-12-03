from config import set_seed, CONFIG
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from model import Model
from test_config import TestConfig
from test_data_loader import create_test_data_loader
import pandas as pd

set_seed(CONFIG.seed)

MODEL_PATH = [
    'FOLD-1.bin',
    'FOLD-2.bin',
    'FOLD-3.bin',
    'FOLD-4.bin',
    'FOLD-5.bin',
]


def validation(model, data_loader, device):
    model.eval()
    predicts = []
    bar = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.no_grad():
        for step, data in bar:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data_loader['mask'].to(device, dtype=torch.long)
            output = model(ids, mask)
            predicts.append(output.view(-1).cpu().detach().numpy())
        predicts = np.concatenate(predicts)
    return predicts


final_predictions = []
for i, path in enumerate(MODEL_PATH):
    model = Model(TestConfig.model_name)
    model.to(TestConfig.device)
    model.load_state_dict(torch.load(path))
    print(f"Prediction for Fold {i + 1}")
    test_data_loader = create_test_data_loader()
    predictions = validation(model, test_data_loader, TestConfig.device)
    final_predictions.append(predictions)

final_predictions = np.array(final_predictions)
final_predictions = np.mean(final_predictions, axis=0)

df = pd.read_csv('jigsaw-toxic-severity-rating/comments_to_score.csv')
df['score'] = final_predictions
df['score'] = final_predictions['score'].rank(method='first')
df.drop('text', axis=1, inplace=True)
df.to_csv('submission.csv', index=False)

