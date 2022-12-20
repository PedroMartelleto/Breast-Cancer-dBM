srun -K \
    --partition=batch \
    --mem-per-cpu=6G \
    --ntasks=1 \
    --cpus-per-task=4 \
    --gpus-per-task=1 \
    --nodes=1 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_22.11-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/netscratch:/netscratch/ \
    python src/create_test_folders.py