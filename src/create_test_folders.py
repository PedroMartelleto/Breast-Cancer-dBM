# Moves benign (x) to benign/benign (x), ...

import os
import globals

DS_PATH = os.path.join(globals.BASE_PATH, "ds/original_ds/Result_DS_test")

# Get class
for root, dirs, files in os.walk(DS_PATH):
    if len(dirs) > 0:
        print("Dataset already split into folders")
        print("Skipping " + root)
        exit(0)

    for file in files:
        print(file)
        if file.endswith(".png"):
            class_name = os.path.basename(file).split(" ")[0]
            print(os.path.join(root, class_name))
            if not os.path.exists(os.path.join(root, class_name)):
                os.mkdir(os.path.join(root, class_name))
            
            os.rename(os.path.join(root, file), os.path.join(root, class_name, file))
