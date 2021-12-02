from config import CONFIG
from model import Model
from transformers import AdamW
from learner import fetch_scheduler, run_training

for fold in range(0, CONFIG.n_fold):
    print(f"__________fold {fold}________________")
    model = Model(CONFIG.model_name)
    model.to(CONFIG.device)
    optimizer = AdamW(model.parameters(), lr=CONFIG.learning_rate, weight_decay=CONFIG.weight_decay)
    scheduler = fetch_scheduler(optimizer)

    model, history = run_training(model, optimizer, scheduler, num_epochs=CONFIG.epochs, fold=fold)
    
