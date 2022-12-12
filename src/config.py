from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Optional, Dict
import os
import globals
import json

@dataclass_json
@dataclass
class ExperimentConfig:
    @staticmethod
    def load(exp_dir):
        with open(os.path.join(globals.BASE_PATH, "experiments", exp_dir, "config.json"), "r") as f:
            ret = ExperimentConfig.from_json(f.read())
            return ret
    
    name: str
    device: str = "cuda:0"
    dst_folder: str = os.path.join(globals.BASE_PATH, "experiments")
    num_epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.001
    gamma: float = 0.1
    step_size: int = 7
    cv_fold: int = 0
    betas: tuple = (0.9, 0.999)
    seed: int = 42
    ds_name: str = "Dataset_BUSI_with_GT"
    ds_num_classes: int = 3

    def save_config(self):
        with open(self.filepath("config.json"), "w") as f:
            f.write(self.to_json(indent=4))
    
    def make_dir_if_necessary(self):
        if not os.path.exists(self.get_folder_path()):
            os.makedirs(self.get_folder_path())

    def filepath(self, filename):
        return os.path.join(self.get_folder_path(), filename)

    def get_folder_path(self):
        return os.path.join(self.dst_folder, self.name)
