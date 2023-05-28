cd ..
srun -K \
    --partition=$1 \
    --mem-per-cpu=5G \
    --ntasks=1 \
    --cpus-per-task=20 \
    --gpus-per-task=1 \
    --nodes=1 \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.03-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts="`pwd`":"`pwd`",/netscratch:/netscratch/ \
    ./install.sh ./scripts/queue_script2.sh $2 $3

# 140*2 + 140*2 + 140 + 140