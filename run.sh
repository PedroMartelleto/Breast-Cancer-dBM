srun -K \
    --partition=A100 \
    --nodes=1 \
    --ntasks=4 \
    --cpus-per-task=10 \
    --gpus-per-task=1 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.11-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/ds:/ds:ro \
    --task-prolog=install_datadings.sh \
    --mem-per-cpu=6G \
    install.sh python src/main.py