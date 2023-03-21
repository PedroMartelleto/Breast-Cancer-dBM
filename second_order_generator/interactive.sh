srun -K \
    --time=00:01:00 \
    --partition=RTX3090 \
    --mem-per-cpu=4G \
    --ntasks=1 \
    --cpus-per-task=4 \
    --gpus-per-task=0 \
    --nodes=1 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.11-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/netscratch:/netscratch/ \
    --pty /bin/bash
./install.sh