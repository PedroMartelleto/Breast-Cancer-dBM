srun -K \
    --time=7:00:00 \
    --partition=batch \
    --mem-per-cpu=14G \
    --ntasks=1 \
    --cpus-per-task=6 \
    --gpus-per-task=1 \
    --nodes=1 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.11-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/netscratch:/netscratch/ \
    --pty /bin/bash

// 192.168.92.189