# Moves benign (x) to benign/benign (x), ...

import os

# Get class
for root, dirs, files in os.walk("/netscratch/martelleto/ultrasound/ds/Dataset_BUSI_with_GT_test"):
    for file in files:
        print(file)
        if file.endswith(".png"):
            class_name = os.path.basename(file).split(" ")[0]
            print(os.path.join(root, class_name))
            if not os.path.exists(os.path.join(root, class_name)):
                os.mkdir(os.path.join(root, class_name))
            
            os.rename(os.path.join(root, file), os.path.join(root, class_name, file))
