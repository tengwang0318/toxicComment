import torch
from transformers import AutoTokenizer


class TestConfig:
    seed = 318
    model_name = "roberta-base"
    test_batch_size = 64
    max_length = 128
    num_classes = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)


