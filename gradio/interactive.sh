srun -K \
    --time=1:00:00 \
    --partition=batch \
    --mem-per-cpu=4G \
    --ntasks=1 \
    --cpus-per-task=15 \
    --gpus-per-task=0 \
    --nodes=1 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.11-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/netscratch:/netscratch/ \
    --pty /bin/bash
./install.sh