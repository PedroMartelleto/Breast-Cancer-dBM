import os
import sys
import numpy as np
from imageio.v3 import imread, imwrite
from skimage.transform import resize
from tqdm import tqdm
import pickle

# For each file in the directories

suffix = ""
INV_MODE = True
DS_PATH = '/netscratch/martelleto/ultrasound/ds/original_ds/Dataset_BUSI_with_GT' + suffix
DESTINATION_PATH = '/netscratch/martelleto/ultrasound/ds/original_ds/Result_DS' + suffix
SIGMA = 16

avg_img = np.zeros((224, 224, 3), dtype=np.float32)
num_images = 0

rects_list = []

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
    global rects_list
    path_repl = '_mask.png'

    mask = load_image(os.path.join(DS_PATH, classname, file.replace('.png', path_repl)))
    row_indices, col_indices, _ = np.where(mask > 0.5)

    if len(row_indices) == 0 or len(col_indices) == 0:
        return cur_img

    min_row = np.min(row_indices)
    max_row = np.max(row_indices)
    min_col = np.min(col_indices)
    max_col = np.max(col_indices)
    width = max_col - min_col + 1
    height = max_row - min_row + 1
    rectangle = (min_col, min_row, width, height)
    # Fill the rectangle with 1s
    rects_list.append(rectangle)
    print(rectangle)
    mask[min_row:max_row+1, min_col:max_col+1, :] = 1

    # Save mask
    return np.multiply(cur_img, 1-mask) + np.multiply(avg_img, mask)
    
    # if not inverse:
    #     return apply_mask(num+1, img_path, np.multiply(cur_img, mask) + np.multiply(avg_img, 1-mask), avg_img, inverse)
    # else:
    #     return apply_mask(num+1, img_path, np.multiply(cur_img, 1-mask) + np.multiply(avg_img, mask), avg_img, inverse)

print("Initializing masker script...")
# print("Calculating average image...")
avg_img[:,:,:] = 0
# for classname in os.listdir(DS_PATH):
#     if not os.path.exists(os.path.join(DESTINATION_PATH, classname)):
#         os.makedirs(os.path.join(DESTINATION_PATH, classname))

#     for file in tqdm(os.listdir(os.path.join(DS_PATH, classname))):
#         if 'mask' in file: continue

#         num_images += 1

#         # Load image
#         img = load_image(os.path.join(DS_PATH, classname, file))
#         mask = load_image(os.path.join(DS_PATH, classname, file.replace('.png', '_mask.png')))
#         mask = gaussian_filter(mask, sigma=SIGMA)
        
#         # Save masked image and inverted masked image
#         if not INV_MODE:
#             img_masked = apply_mask(0, os.path.join(DS_PATH, classname, file), img, avg_img, inverse=False)
#         else:
#             img_masked = apply_mask(0, os.path.join(DS_PATH, classname, file), img, avg_img, inverse=True)
        
        # Rolling average using avg_img
        # avg_img is black for now
        # avg_img = (avg_img * (num_images - 1) + img_masked) / num_images

# Save average image
# save_image(os.path.join(DESTINATION_PATH, 'avgimg' + suffix + '.png'), avg_img)

print("Applying masks...")
for classname in os.listdir(DS_PATH):
    if not os.path.exists(os.path.join(DESTINATION_PATH, classname)):
        os.makedirs(os.path.join(DESTINATION_PATH, classname))

    for file in tqdm(os.listdir(os.path.join(DS_PATH, classname))):
        if 'mask' in file:
            continue

        # Load image
        try:
            img = load_image(os.path.join(DS_PATH, classname, file))
            mask = load_image(os.path.join(DS_PATH, classname, file.replace('.png', '_mask.png')))
        except Exception as e:
            print(e)
            continue
        # mask = gaussian_filter(mask, sigma=SIGMA)
        #save_image(os.path.join(DESTINATION_PATH, classname, file.replace('.png', '_mask.png')), mask)
        
        # Save masked image and inverted masked image
        # if not INV_MODE:
        #     img_masked = apply_mask(0, os.path.join(DS_PATH, classname, file), img, avg_img, inverse=False)
        #     save_image(os.path.join(DESTINATION_PATH, classname, file.replace('.png', '_mult.png')), img_masked)
        # else:
        inv_img_masked = apply_mask(0, os.path.join(DS_PATH, classname, file), img, avg_img, inverse=True)
        save_image(os.path.join(DESTINATION_PATH, classname, file.replace('.png', '_inv_mult.png')), inv_img_masked)

# Save rectangles list to file
with open(os.path.join(DESTINATION_PATH, 'rects_list' + suffix + '.pkl'), 'wb') as f:
    pickle.dump(rects_list, f)