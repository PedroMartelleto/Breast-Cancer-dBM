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

# NOTES

- Dataset sizes