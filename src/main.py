from config import ExperimentConfig
from train import TrainHelper, OptimContextFactory
from model import ModelFactory
import torch
from datasets import DatasetWrapper
import os, uuid, hyper, globals, json, ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from explainability import Explainer
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader
from ray.tune.trainable.session import get_trial_id
from datasets import AugDataset
from tqdm import tqdm

def rand_uuid():
    return str(uuid.uuid4())[:8]

def run_experiment(hyper_config, exp_name, fold, ray_tune=False, seed=42, imageNet=True):
    exp = ExperimentConfig(name=exp_name + ((get_trial_id()) if ray_tune else ""), 
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
                           seed = seed,
                           ds_name = "original_ds/INV_MASKED_Dataset_BUSI_with_GT",
                           ds_num_classes = 3)
    exp.make_dir_if_necessary()
    exp.save_config()

    device = torch.device(exp.device)

    model = ModelFactory.create_model(device, num_classes=exp.ds_num_classes, imageNet=imageNet)
    ds = DatasetWrapper(os.path.join("ds", exp.ds_name), exp)

    optim_context = OptimContextFactory.create_optimizer(model, exp)

    trainer = TrainHelper(device=device, ds=ds, exp_config=exp,
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

    num_samples = 40

    print("Running {} trials...".format(num_samples))
    result = tune.run(
        partial(run_experiment, exp_name="imagenet-tune", fold=0, ray_tune=True, imageNet=True),
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

def create_exp(exp_name, hyper_config, fold):
    return ExperimentConfig(name=exp_name, 
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

def calc_confusion_matrix(exp_name, fold):
    hyper_config = hyper.get_tuned_hyperparams()
    
    exp = create_exp(exp_name, hyper_config, fold)
    device = torch.device(exp.device)
    model = ModelFactory.create_model_from_checkpoint(exp_name, device, num_classes=exp.ds_num_classes)
    ds = DatasetWrapper(os.path.join("ds", exp.ds_name), exp)

    trainer = TrainHelper(device=device, ds=ds, exp_config=exp,
                          model=model, optim_context=[None, None, None], ray_tune=False)
    trainer.save_confusion_matrix()

# bash script one-liner that removes all folders with only one file in them named "config.json"
# shopt -s globstar; for d in **/; do f=("$d"/*); [[ ${#f[@]} -eq 1 && -f "$f" && "${f##*/}" =~ ^config.json$ ]] && rm -r -- "$d"; done

def interpret_model():
    exp = create_exp(globals.RANDOM_INIT_EXP_NAMES[0], hyper.get_tuned_hyperparams(), 0)
    device = torch.device(exp.device)
    model = ModelFactory.create_model_from_checkpoint(globals.RANDOM_INIT_EXP_NAMES[0], "cuda:0", num_classes=exp.ds_num_classes)
    ds = DatasetWrapper(os.path.join("ds", exp.ds_name), exp)

    explainer = Explainer(model, device, ds)
    prefix = "/netscratch/martelleto/ultrasound/explain/"

    # Iterate over images in test set folder
    for root, dirs, files in os.walk(globals.TEST_DS_PATH):
        for file in tqdm(files):
            img = Image.open(os.path.join(root, file))
            explainer.shap(img, os.path.join(prefix, "SHAP_" + file))
            explainer.occlusion(img, os.path.join(prefix, "OCC_" + file))
            explainer.noise_tunnel(img, os.path.join(prefix, "NT_" + file))
            explainer.gradcam(img, os.path.join(prefix, "GRAD_" + file))

def random_inits():
    hyper_config = hyper.get_tuned_hyperparams("nofinetune_tune_hypers")
    
    # Use ray to train multiple experiments in parallel with different seeds
    for seed in globals.SEEDS[0:1]:
        run_experiment(hyper_config, exp_name=f"IGNORE_____noimagenet-random-{seed}", fold=0, seed=seed, ray_tune=False)

def calc_conf_matrix_for_exp(exp_name, ds):
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    best_model = ModelFactory.create_model_from_checkpoint(exp_name, device, num_classes=3)
    class_names = list(ds.class_to_idx.keys())
    test_loader = DataLoader(ds, batch_size=128)
    exp = create_exp(exp_name, hyper.get_tuned_hyperparams("nofinetune_tune_hypers"), 0)

    trainer = TrainHelper(device=device, ds=None, exp_config=exp,
                          model=best_model, optim_context=[None, None, None], ray_tune=False)
    trainer.save_confusion_matrix(test_loader, exp_name, class_names)

#def remove_first_order_feats():

if __name__ == "__main__":
    # 1 - tune hyperparameters
    #tune_hyperparameters()

    # 2 - random inits
    #random_inits()

    # 3 - calculate confusion matrices from repeated experiments
    #transform = transforms.Compose([ transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=globals.NORM_MEAN, std=globals.NORM_STD) ])
    #for seed in globals.SEEDS:
    #    ds = AugDataset(globals.TEST_DS_PATH, aug=None, transform=transform)
    #    calc_conf_matrix_for_exp(f"noimagenet-random-{seed}", ds)
    
    # 4 - See ConfMatrix.ipynb

    # 5 - use captum for model interpretation
    # interpret_model()

    # 6 - deploy to HF spaces & gradio.app

    # https://captum.ai/tutorials/Resnet_TorchVision_Interpret