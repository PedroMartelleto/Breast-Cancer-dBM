# Docs

1. Data augmentation
        A.Resize(256, 256),
        A.ShiftScaleRotate(shift_limit=0.008, scale_limit=0.2, rotate_limit=30, p=0.7),
        A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, p=0.5),
        A.Sharpen(alpha=(0.05, 0.1), lightness=(0.9, 1.1), p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        A.PadIfNeeded(224, 224),
        A.RandomCrop(width=224, height=224),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD)


2. Hyperparameter tuning (40 random samples)
        search_space = {
            "learning_rate": tune.loguniform(1e-4, 1e-1),
            "momentum": tune.uniform(0.9, 0.99),
            "batch_size": tune.choice([8, 16, 32, 64]),
            "step_size": tune.choice([5, 10, 15, 20]),
            "num_epochs": tune.choice([10, 20, 30, 40, 50, 60]),
            "gamma": tune.uniform(0.1, 0.9)
        }

3. Testing on test set with 10 seeds and re-shuffling
        Conf_Matrix = [
            [],
            [],
            []
        ]

3.5. Test 2 and 3 with and without imagenet pre-training

4. Captum on all test images

5. HF Deploy & front-end

6. We're done :)