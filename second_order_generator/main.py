import os
import sys
import numpy as np
from imageio.v3 import imread, imwrite
from skimage.transform import resize
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# For each file in the directories

suffix = ""
INV_MODE = True
DS_PATH = '/netscratch/martelleto/ultrasound/ds/original_ds/Dataset_BUSI_with_GT' + suffix
DESTINATION_PATH = '/netscratch/martelleto/ultrasound/ds/original_ds/INV_MASKED_Dataset_BUSI_with_GT' + suffix
SIGMA = 8

avg_img = np.zeros((224, 224, 3), dtype=np.float32)
num_images = 0

if not os.path.exists(DESTINATION_PATH):
    os.makedirs(DESTINATION_PATH)

def load_image(img_path):
    img = imread(img_path)

    if len(img.shape) == 3:
        img = img[:,:,0]

    if img.dtype == np.bool8:
        mx = img.max()
        img = img.astype(np.float32) / (mx if mx > 0 else 1.0)
    else:
        img = img.astype(np.float32) / 255.0

    img = np.stack((img, img, img), axis=2)
    img = resize(img, (224, 224), order=1)

    return img

def save_image(img_path, img):
    imwrite(img_path, (img*255).astype(np.uint8), format="png", compress_level=0)

def apply_mask(num, img_path, cur_img, avg_img, inverse=False):
    path_repl = '_mask.png'

    if num > 0:
        path_repl = '_mask_' + str(num) + '.png'

    try:
        mask = load_image(os.path.join(DS_PATH, classname, file.replace('.png', path_repl)))
    except:
        return cur_img
    
    if not inverse:
        return apply_mask(num+1, img_path, np.multiply(cur_img, mask) + np.multiply(avg_img, 1-mask), avg_img, inverse)
    else:
        return apply_mask(num+1, img_path, np.multiply(cur_img, 1-mask) + np.multiply(avg_img, mask), avg_img, inverse)

print("Initializing masker script...")
print("Calculating average image...")

for classname in os.listdir(DS_PATH):
    if not os.path.exists(os.path.join(DESTINATION_PATH, classname)):
        os.makedirs(os.path.join(DESTINATION_PATH, classname))

    for file in tqdm(os.listdir(os.path.join(DS_PATH, classname))):
        if 'mask' in file:
            continue

        num_images += 1

        # Load image
        img = load_image(os.path.join(DS_PATH, classname, file))

        # rolling average
        avg_img = (avg_img * (num_images - 1) + img) / num_images

# Save average image
save_image(os.path.join(DESTINATION_PATH, 'avgimg' + suffix + '.png'), avg_img)

print("Applying masks...")
for classname in os.listdir(DS_PATH):
    if not os.path.exists(os.path.join(DESTINATION_PATH, classname)):
        os.makedirs(os.path.join(DESTINATION_PATH, classname))

    for file in tqdm(os.listdir(os.path.join(DS_PATH, classname))):
        if 'mask' in file:
            continue

        num_images += 1

        # Load image
        img = load_image(os.path.join(DS_PATH, classname, file))
        mask = load_image(os.path.join(DS_PATH, classname, file.replace('.png', '_mask.png')))
        mask = gaussian_filter(mask, sigma=SIGMA)
        #save_image(os.path.join(DESTINATION_PATH, classname, file.replace('.png', '_mask.png')), mask)
        
        # Save masked image and inverted masked image
        if not INV_MODE:
            img_masked = apply_mask(0, os.path.join(DS_PATH, classname, file), img, avg_img, inverse=False)
            save_image(os.path.join(DESTINATION_PATH, classname, file.replace('.png', '_mult.png')), img_masked)
        else:
            inv_img_masked = apply_mask(0, os.path.join(DS_PATH, classname, file), img, avg_img, inverse=True)
            save_image(os.path.join(DESTINATION_PATH, classname, file.replace('.png', '_inv_mult.png')), inv_img_masked)