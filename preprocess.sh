srun -K \
    -w gera \
    --partition=V100-32GB \
    --mem-per-cpu=12G \
    --ntasks=1 \
    --cpus-per-task=1 \
    --gpus-per-task=1 \
    --nodes=1 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.11-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/netscratch:/netscratch/ \
    install.sh python src/preprocess.py