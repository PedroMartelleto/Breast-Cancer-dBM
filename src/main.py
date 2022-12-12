from config import ExperimentConfig
from train import TrainHelper, OptimContextFactory
from model import ModelFactory
import torch
from datasets import DatasetWrapper
import os
import hyper
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import uuid
import globals
import json

# TODO: Distributed training?

def rand_uuid():
    return str(uuid.uuid4())[:8]

def run_experiment(tune_config, exp_name, fold):
    exp = ExperimentConfig(name=exp_name + "-" + rand_uuid(), 
                            device="cuda:0" if torch.cuda.is_available() else "cpu",
                            num_epochs = tune_config["num_epochs"],
                            batch_size = tune_config["batch_size"],
                            learning_rate = tune_config["learning_rate"],
                            momentum = tune_config["momentum"],
                            weight_decay = 0.0001,
                            gamma = tune_config["gamma"],
                            step_size = tune_config["step_size"],
                            cv_fold = 0,
                            betas = (0.9, 0.999),
                            seed = 42,
                            ds_name = "Dataset_BUSI_with_GT",
                            ds_num_classes = 3)
    exp.make_dir_if_necessary()

    device = torch.device(exp.device)

    model = ModelFactory.create_model(device, num_classes=exp.ds_num_classes)
    ds = DatasetWrapper(os.path.join("ds", exp.ds_name), exp)

    optim_context = OptimContextFactory.create_optimizer(model, exp)

    trainer = TrainHelper(fold=fold, device=device, ds=ds, exp_config=exp,
                          model=model, optim_context=optim_context)
    trainer.train_and_validate()

def tune_hyperparameters():
    print("Preparing to tune hyperparameters...")

    scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=60,
            grace_period=1,
            reduction_factor=2)

    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    num_samples = 30

    print("Running {} trials...".format(num_samples))
    result = tune.run(
        partial(run_experiment, exp_name="tune-hyp", fold=0),
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=hyper.search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        checkpoint_at_end=True)
    
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))

    with open("best_config.json", "w") as f:
        json.dump(best_trial.config, f)
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))

if __name__ == "__main__":
    tune_hyperparameters()