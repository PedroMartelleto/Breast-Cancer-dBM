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
SRC_PATH = '/netscratch/martelleto/ultrasound/ds/original_ds/Result_DS' + suffix
DESTINATION_PATH = '/netscratch/martelleto/ultrasound/ds/original_ds/Result_DS' + suffix
SIGMA = 16

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
# Save rectangles list to file
with open(os.path.join(SRC_PATH, 'rects_list' + suffix + '.pkl'), 'rb') as f:
    rects_list = pickle.load(f)

xs = []
ys = []
ws = []
hs = []

for rect in rects_list:
    x, y, w, h = rect
    xs.append(x)
    ys.append(y)
    ws.append(w)
    hs.append(h)

xs = np.array(xs)
ys = np.array(ys)
ws = np.array(ws)
hs = np.array(hs)

xs_mean = xs.mean()
ys_mean = ys.mean()
ws_mean = ws.mean()
hs_mean = hs.mean()

xs_std = xs.std()
ys_std = ys.std()
ws_std = ws.std()
hs_std = hs.std()

classname = 'normal'

if not os.path.exists(os.path.join(DESTINATION_PATH, classname)):
    os.makedirs(os.path.join(DESTINATION_PATH, classname))

for file in tqdm(os.listdir(os.path.join(SRC_PATH, classname))):
    if 'mask' in file:
        continue

    # Load image
    try:
        img = load_image(os.path.join(SRC_PATH, classname, file))
    except Exception as e:
        print(e)
        continue

    # Sample from rects_list
    x = int(np.random.normal(xs_mean, xs_std))
    y = int(np.random.normal(ys_mean, ys_std))
    w = int(np.random.normal(ws_mean, ws_std))
    h = int(np.random.normal(hs_mean, hs_std))
    
    x = min(max(0, x), img.shape[1] - 2)
    y = min(max(0, y), img.shape[0] - 2)
    w = min(max(1, w), img.shape[1] - x - 1)
    h = min(max(1, h), img.shape[0] - y - 1)

    print(x, y, w, h)

    # Fill the rectangle with black
    img[y:y+h, x:x+w, :] = 0

    # Save new image
    save_image(os.path.join(DESTINATION_PATH, classname, file.replace('.png', '_rect.png')), img)