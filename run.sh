srun -K \
    --partition=A100 \
    --mem-per-cpu=12G \
    --ntasks=4 \
    --cpus-per-task=1 \
    --gpus-per-task=1 \
    --nodes=4 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.11-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/netscratch:/netscratch/ \
    install.sh python src/main.py