import json
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Optional
import os

@dataclass_json
@dataclass
class TrainConfig:
    name: str
    num_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.001
    gamma: float = 0.1
    step_size: int = 7
    train_fold: int = 0
    betas: tuple = (0.9, 0.999)
    seed: int = 42

@dataclass_json
@dataclass
class ExperimentConfig:
    name: str
    device: str = "cuda:0"
    train_configs: List = field(default_factory=list)
    dst_folder: str = "/netscratch/martelleto/"

    def add_train_config(self, **kwargs):
        train_config = TrainConfig("cv_" + str(kwargs["train_fold"]), **kwargs)
        self.train_configs.append(train_config)
        return train_config
    
    def make_dir_if_necessary(self):
        if not os.path.exists(self.get_folder_path()):
            os.makedirs(self.get_folder_path())

    def filepath(self, filename):
        return os.path.join(self.get_folder_path(), filename)

    def get_folder_path(self):
        return os.path.join(self.dst_folder, self.name)
