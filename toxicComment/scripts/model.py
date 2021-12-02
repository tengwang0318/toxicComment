from transformers import AutoModel, AutoTokenizer, AdamW
import torch.nn as nn
import torch
import numpy as np

from config import set_seed, CONFIG

set_seed()


class Model(nn.Module):
    def __init__(self, model_name):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(768, CONFIG.n_classes)

    def forward(self, ids, mask):
        out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
        out = self.drop(out[1])
        outputs = self.fc(out)
        return outputs

