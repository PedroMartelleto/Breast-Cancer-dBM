import os
import globals
from tqdm import tqdm
import shutil

# NOTE: Both datasets are the same!!!!

DS_PATH = os.path.join(globals.BASE_PATH, "ds/")

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def prepare_Dataset_BUSI_with_GT():
    if os.path.exists(os.path.join(DS_PATH, "Dataset_BUSI_with_GT_masks/")):
        print("Dataset_BUSI_with_GT already preprocessed. Skipping...")
        return

    def move_masks():
        src_path = os.path.join(DS_PATH, "Dataset_BUSI_with_GT/")
        dst_path = os.path.join(DS_PATH, "Dataset_BUSI_with_GT_masks/")

        # Create the destination folder if it doesn't exist
        create_dir_if_not_exists(dst_path)

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
    
    move_masks()

def move_all_files_in_dir(src, dst):
    for root, dirs, files in os.walk(src):
        for file in files:
            shutil.move(os.path.join(root, file), dst)

def prepare_sharam_ds():
    if os.path.exists(os.path.join(DS_PATH, "binary_ds/")):
        print("Binary DS already preprocessed. Skipping...")
        return

    src_path = os.path.join(DS_PATH, "ultrasound breast classification/")

    # Rename folder from 'ultrasound breast classification' to 'binary_ds',
    # if 'ultrasound breast classification' exists
    os.rename(src_path, os.path.join(DS_PATH, "binary_ds/"))

    os.makedirs(os.path.join(DS_PATH, "binary_ds/malignant/"))
    os.makedirs(os.path.join(DS_PATH, "binary_ds/benign/"))

    # Move all files from 'binary_ds' to 'binary_ds/malignant' and 'binary_ds/benign'
    move_all_files_in_dir(os.path.join(DS_PATH, "binary_ds/train/malignant"),
                          os.path.join(DS_PATH, "binary_ds/malignant/"))
    move_all_files_in_dir(os.path.join(DS_PATH, "binary_ds/val/malignant"),
                          os.path.join(DS_PATH, "binary_ds/malignant/"))

    move_all_files_in_dir(os.path.join(DS_PATH, "binary_ds/train/benign"),
                          os.path.join(DS_PATH, "binary_ds/benign/"))
    move_all_files_in_dir(os.path.join(DS_PATH, "binary_ds/val/benign"),
                          os.path.join(DS_PATH, "binary_ds/benign/"))
    
    # Remove empty folders
    os.rmdir(os.path.join(DS_PATH, "binary_ds/train/malignant/"))
    os.rmdir(os.path.join(DS_PATH, "binary_ds/train/benign/"))
    os.rmdir(os.path.join(DS_PATH, "binary_ds/val/malignant"))
    os.rmdir(os.path.join(DS_PATH, "binary_ds/val/benign/"))

    os.rmdir(os.path.join(DS_PATH, "binary_ds/train/"))
    os.rmdir(os.path.join(DS_PATH, "binary_ds/val/"))

def join_datasets():
    # Create the destination folder if it doesn't exist
    if not os.path.exists(os.path.join(DS_PATH, "joined_ds/")):
        os.makedirs(os.path.join(DS_PATH, "joined_ds/"))

    # Move non-augmented from binary_ds to joined dataset
    for root, dirs, files in os.walk(os.path.join(DS_PATH, "binary_ds/")):
        for file in files:
            if file.endswith(").png") and "-" not in file:
                dirname = root.split(os.path.sep)[-1]
                dst = os.path.join(DS_PATH, "joined_ds/", dirname)
                create_dir_if_not_exists(dst)
                shutil.copyfile(os.path.join(root, file), os.path.join(dst, "ds0_" + file))

    # Move all files from Dataset_BUSI_with_GT to joined dataset
    for root, dirs, files in os.walk(os.path.join(DS_PATH, "Dataset_BUSI_with_GT/")):
        for file in files:
            if file.endswith(".png") and not "mask" in file:
                dirname = root.split(os.path.sep)[-1]
                dst = os.path.join(DS_PATH, "joined_ds/", dirname)
                create_dir_if_not_exists(dst)
                shutil.copyfile(os.path.join(root, file), os.path.join(dst, "ds1_" + file))

prepare_Dataset_BUSI_with_GT()
#prepare_sharam_ds()
#join_datasets()