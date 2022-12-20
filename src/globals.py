import os

BASE_PATH = "/netscratch/martelleto/ultrasound"

# Because we are fine-tuning from ImageNet...
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

DS_PATH = os.path.join(BASE_PATH, "ds", "Dataset_BUSI_with_GT")
TEST_DS_PATH = os.path.join(BASE_PATH, "ds", "Dataset_BUSI_with_GT_test")

SEEDS = [8974274, 6859559, 917234, 8458981, 3240563, 7103588, 5232396, 3991434, 2158036, 5012237]
RANDOM_INIT_EXP_NAMES = [f"NEW-random-{seed}" for seed in SEEDS]