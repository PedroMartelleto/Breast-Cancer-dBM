import os
import globals
from tqdm import tqdm

DS_PATH = os.path.join(globals.BASE_PATH, "ds/")

src_path = os.path.join(DS_PATH, "Dataset_BUSI_with_GT/")
dst_path = os.path.join(DS_PATH, "Dataset_BUSI_with_GT_masks/")

# Create the destination folder if it doesn't exist
if not os.path.exists(dst_path):
    os.makedirs(dst_path)

# Goes through all the files recursively
for root, dirs, files in os.walk(src_path):
    for file in tqdm(files):
        # If the file is a mask
        if "mask" in file or "maks" in file:
            # Get full file path
            file_path = os.path.join(root, file)
            dir = os.path.dirname(root)
            # Move it to the destination folder

            if not os.path.exists(dst_path):
                os.makedirs(os.path.join(dst_path, dir))

            os.rename(file_path, os.path.join(dst_path, dir, file))