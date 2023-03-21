import os
import sys
import numpy as np
from imageio.v3 import imread, imwrite
from skimage.transform import resize
from tqdm import tqdm
import create_test_ds
import create_test_folders

# For each file in the directories

def random_rect_gen():
    create_test_ds.run()
    create_test_folders.run()

    for suffix in ["", "_test"]:
        INV_MODE = True
        SRC_PATH = '/netscratch/martelleto/ultrasound/ds/Result_DS' + suffix
        DESTINATION_PATH = '/netscratch/martelleto/ultrasound/ds/Result_DS' + suffix
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

        SIZES_JOINT_MEAN = np.array([1.43174167e+00, 6.80290521e+03])
        SIZES_JOINT_COV = np.array([[ 3.40688872e-01, -3.42400038e+02],
                                    [-3.42400038e+02,  5.11123773e+07]])

        XY_JOINT_MEAN = np.array([67.60505529, 44.79462875])
        XY_JOINT_COV = np.array([[1796.72984782, 179.04850821],
                                [ 179.04850821, 639.407123  ]])

        classname = 'normal'

        listdir_normal = os.listdir(os.path.join(SRC_PATH, classname))
        num_rects_to_generate = len(listdir_normal)
        size_samples = np.random.multivariate_normal(SIZES_JOINT_MEAN, SIZES_JOINT_COV, size=num_rects_to_generate)
        xy_samples = np.random.multivariate_normal(XY_JOINT_MEAN, XY_JOINT_COV, size=num_rects_to_generate)

        # Overwrite rectangles with new distribution?????

        if not os.path.exists(os.path.join(DESTINATION_PATH, classname)):
            os.makedirs(os.path.join(DESTINATION_PATH, classname))

        i = 0

        for file in tqdm(listdir_normal):
            if 'mask' in file:
                continue

            # Load image
            try:
                img = load_image(os.path.join(SRC_PATH, classname, file))
            except Exception as e:
                print(e)
                continue

            # Sample from rects_list
            x = int(xy_samples[i, 0])
            y = int(xy_samples[i, 1])
            aspect_ratio = max(int(size_samples[i, 0]), 0.01)
            area = max(int(size_samples[i, 1]), 1)
            i += 1

            w = int(np.sqrt(max(area * aspect_ratio, 1)))
            h = int(np.sqrt(max(area / aspect_ratio, 1)))

            # ensure the rectangle is not too much to the border
            x = min(x, int(img.shape[1] - np.random.uniform(32, 50) - 1))
            y = min(y, int(img.shape[0] - np.random.uniform(32, 50) - 1))

            x = min(max(0, x), img.shape[1] - 2)
            y = min(max(0, y), img.shape[0] - 2)
            w = min(max(int(np.random.uniform(32, 96)), w), img.shape[1] - x - 1)
            h = min(max(int(np.random.uniform(32, 96)), h), img.shape[0] - y - 1)

            print(x, y, w, h)

            # Fill the rectangle with black
            img[y:y+h, x:x+w, :] = 0

            # Save new image
            save_image(os.path.join(DESTINATION_PATH, classname, file.replace('.png', '_rect.png')), img)

if __name__ == '__main__':
    random_rect_gen()