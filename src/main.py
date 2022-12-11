from config import TrainConfig, ExperimentConfig
from train import TrainHelper, OptimContextFactory
from model import ModelFactory
import torch
from datasets import DatasetWrapper

exp = ExperimentConfig("simple")
exp.add_train_config(num_epochs=30, batch_size=64, cv_fold=0)
exp.make_dir_if_necessary()

with open(exp.filepath("config.json"), "w") as f:
    f.write(exp.to_json(indent=4))

device = torch.device(exp.device)

model = ModelFactory.create_model(device)
optim_context = OptimContextFactory.create_optimizer(model, exp.train_configs[0])

ds = DatasetWrapper("ds/Dataset_BUSI_with_GT", exp.train_configs[0])

# Trains the model!
trainer = TrainHelper(device, ds, exp, exp.train_configs[0], model, *optim_context)
trainer.train()