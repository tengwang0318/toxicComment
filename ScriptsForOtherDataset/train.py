from config import CONFIG
from model import Model
from transformers import AdamW
from learner import fetch_scheduler, run_training

model = Model(CONFIG.model_name)
model.to(CONFIG.device)
optimizer = AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
scheduler = fetch_scheduler(optimizer)

model, history = run_training(model, optimizer, scheduler, num_epochs=CONFIG.epochs)
