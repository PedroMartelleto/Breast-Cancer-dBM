srun -K \
    --partition=RTX2080Ti \
    --mem-per-cpu=16G \
    --ntasks=1 \
    --cpus-per-task=4 \
    --gpus-per-task=1 \
    --nodes=1 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.11-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/netscratch:/netscratch/ \
    install.sh python random_rect_gen.py