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

def rand_uuid():
    return str(uuid.uuid4())[:8]

def run_experiment(hyper_config, exp_name, fold, ray_tune=False):
    exp = ExperimentConfig(name=exp_name + "-" + rand_uuid(), 
                            device="cuda:0" if torch.cuda.is_available() else "cpu",
                            num_epochs = hyper_config["num_epochs"],
                            batch_size = hyper_config["batch_size"],
                            learning_rate = hyper_config["learning_rate"],
                            momentum = hyper_config["momentum"],
                            weight_decay = 0.0001,
                            gamma = hyper_config["gamma"],
                            step_size = hyper_config["step_size"],
                            cv_fold = fold,
                            betas = (0.9, 0.999),
                            seed = 42,
                            ds_name = "Dataset_BUSI_with_GT",
                            ds_num_classes = 3)
    exp.make_dir_if_necessary()
    exp.save_config()

    device = torch.device(exp.device)

    model = ModelFactory.create_model(device, num_classes=exp.ds_num_classes)
    ds = DatasetWrapper(os.path.join("ds", exp.ds_name), exp)

    optim_context = OptimContextFactory.create_optimizer(model, exp)

    trainer = TrainHelper(fold=fold, device=device, ds=ds, exp_config=exp,
                          model=model, optim_context=optim_context, ray_tune=ray_tune)
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
        partial(run_experiment, exp_name="tune-hyp", fold=0, ray_tune=True),
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

def calc_confusion_matrix(exp_name, fold):
    hyper_config = hyper.get_tuned_hyperparams()
    exp = ExperimentConfig(name=exp_name, 
                            device="cuda:0" if torch.cuda.is_available() else "cpu",
                            num_epochs = hyper_config["num_epochs"],
                            batch_size = hyper_config["batch_size"],
                            learning_rate = hyper_config["learning_rate"],
                            momentum = hyper_config["momentum"],
                            weight_decay = 0.0001,
                            gamma = hyper_config["gamma"],
                            step_size = hyper_config["step_size"],
                            cv_fold = fold,
                            betas = (0.9, 0.999),
                            seed = 42,
                            ds_name = "Dataset_BUSI_with_GT",
                            ds_num_classes = 3)
    device = torch.device(exp.device)
    model = ModelFactory.create_model_from_checkpoint(exp_name, device, num_classes=exp.ds_num_classes)
    ds = DatasetWrapper(os.path.join("ds", exp.ds_name), exp)

    trainer = TrainHelper(fold=fold, device=device, ds=ds, exp_config=exp,
                          model=model, optim_context=[None, None, None], ray_tune=False)
    trainer.save_confusion_matrix()

if __name__ == "__main__":
    # 1 - tune hyperparameters
    # tune_hyperparameters()

    # 2 - train cross-validation models
    #for fold in range(5):
    #    run_experiment(hyper.get_tuned_hyperparams(), "tuned-model-cv{}".format(fold), fold, ray_tune=False)

    # 3 - calculate confusion matrices from each cross-validation model
    #for fold in range(5):
    #    calc_confusion_matrix(globals.CV_EXP_NAMES[fold], fold)
    
    # 4 - See ConfMatrix.ipynb

    # 5 - use captum for model interpretation
    #for fold in range(5):
    # https://captum.ai/tutorials/Resnet_TorchVision_Interpret