import os
import globals
import torch

DS_PATH = os.path.join(globals.BASE_PATH, "ds/original_ds/INV_MASKED_Dataset_BUSI_with_GT")
if DS_PATH.endswith("/"): DS_PATH = DS_PATH[:-1]
TEST_DS_PATH = DS_PATH + "_test"

# count files recursively in DS_PATH
imgs = []
ds_size = 0

for root, dirs, files in os.walk(DS_PATH):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            imgs.append(os.path.join(root, file))
    ds_size += len(files)

print(f"DS size: {ds_size}")

test_ds_size = int(0.2 * ds_size)
train_ds_size = ds_size - test_ds_size

gen = torch.Generator()
gen.manual_seed(32)
_, ds_test = torch.utils.data.random_split(imgs, [train_ds_size, test_ds_size])

if not os.path.exists(TEST_DS_PATH):
    os.mkdir(TEST_DS_PATH)
    for i in ds_test:
        os.rename(i, os.path.join(TEST_DS_PATH, os.path.basename(i)))
else:
    print("test folder already exists")