--mail-user=martelleto.pedro@gmail.com
 --mail-type=ALL,TIME_LIMIT_90
touch ~/.logoptin # for logging

$ srun -K \
  --job-name="Test" \
  --gpus=1 \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
  --container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh \
  --container-workdir="`pwd`" \
  python main.py

# TODO

- Captum with the results of hypertuning https://captum.ai/tutorials/Resnet_TorchVision_Interpret
- Cross-validation for final results: train helper for loop, confusion matrix
- REST API with final nn

# NOTES

## Datasets

### Sharam DS https://www.kaggle.com/datasets/vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer

### 2k DS https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset

# Caveats

* Both datasets are (very) small
* Sharam DS is already augumented, but we ignore the augmentations and only look at the original image
* Both datasets are equal!
