cd ..
srun -K \
    --time=30:00:00 \
    --partition=$1 \
    --mem-per-cpu=5G \
    --ntasks=1 \
    --cpus-per-task=20 \
    --gpus-per-task=1 \
    --nodes=1 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.03-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/netscratch:/netscratch/ \
    --pty /bin/bash