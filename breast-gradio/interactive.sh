srun -K \
    --time=01:00:00 \
    --partition=RTX3090 \
    --mem-per-cpu=14G \
    --ntasks=1 \
    --cpus-per-task=2 \
    --gpus-per-task=1 \
    --nodes=1 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.11-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/netscratch:/netscratch/ \
    --pty /bin/bash