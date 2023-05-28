cd ..
srun -K \
    --time=24:00:00 \
    --partition=A100-40GB \
    --mem-per-cpu=2G \
    --ntasks=1 \
    --cpus-per-task=30 \
    --gpus-per-task=0 \
    --nodes=1 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.04-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/netscratch:/netscratch/ \
    --pty /bin/bash