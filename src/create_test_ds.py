import os
import globals
import torch

DS_PATH = os.path.join(globals.BASE_PATH, "ds", "Dataset_BUSI_with_GT")

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

if not os.path.exists(globals.TEST_DS_PATH):
    os.mkdir(globals.TEST_DS_PATH)
    for i in ds_test:
        os.rename(i, os.path.join(globals.TEST_DS_PATH, os.path.basename(i)))
else:
    print("test folder already exists")