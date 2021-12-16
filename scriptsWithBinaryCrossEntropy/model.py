from transformers import AutoModel, AutoTokenizer, AdamW
import torch.nn as nn
import torch
import numpy as np

from config import set_seed, CONFIG

set_seed(CONFIG.seed)


class Model(nn.Module):
    def __init__(self, model_name):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768 * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ids1, mask1, ids2, mask2):
        out1 = self.model(input_ids=ids1, attention_mask=mask1, output_hidden_states=False)
        out1 = self.drop(out1[1])
        # outputs1 = self.fc(out1)
        out2 = self.model(input_ids=ids2, attention_mask=mask2, output_hidden_states=False)
        out2 = self.drop(out2[1])
        out = torch.cat([out1, out2], dim=1)
        outputs = self.fc(out)
        outputs = self.sigmoid(outputs)
        # outputs2 = self.fc(out2)
        return outputs
