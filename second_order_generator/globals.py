import os

BASE_PATH = "/netscratch/martelleto/ultrasound"

# Because we are fine-tuning from ImageNet...
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

DS_NAME = "original_ds/INV_MASKED_Dataset_BUSI_with_GT"

DS_PATH = os.path.join(BASE_PATH, "ds", DS_NAME)
TEST_DS_PATH = os.path.join(BASE_PATH, "ds", DS_NAME + "_test")

SEEDS = [8974274, 6859559, 917234, 8458981, 3240563, 7103588, 5232396, 3991434, 2158036, 5012237]