from ray import tune
import numpy as np
import json

search_space = {
    "learning_rate": tune.loguniform(1e-4, 1e-1),
    "momentum": tune.uniform(0.9, 0.99),
    "batch_size": tune.choice([8, 16, 32, 64]),
    "step_size": tune.choice([5, 10, 15, 20]),
    "num_epochs": tune.choice([10, 20, 30, 40, 50, 60]),
    "gamma": tune.uniform(0.1, 0.9)
}

def get_tuned_hyperparams():
    with open("hypertuning/best_config.json", "r") as f:
        best_config = json.load(f)
        return best_config