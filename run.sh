./deploy_experiments.sh
srun -K \
    --partition=RTXA6000 \
    --mem-per-cpu=6G \
    --ntasks=1 \
    --cpus-per-task=3 \
    --gpus-per-task=1 \
    --nodes=1 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.11-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/netscratch:/netscratch/ \
    install.sh python src/main.py