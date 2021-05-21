TXT_DB=$1
IMG_DIR=$2
OUTPUT=$3
PRETRAIN_DIR=$4
# TXT_RAW=$6

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

# if [ "$5" = "--prepro" ]; then
#     RO=""
# else
#     RO=",readonly"
# fi


# docker run --gpus "device=$CUDA_VISIBLE_DEVICES" --ipc=host --rm -it \
# -e NCCL_IB_CUDA_SUPPORT=0 \



docker run -p 8892:8892 --runtime=nvidia --ipc=host --rm -it \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$OUTPUT,dst=/storage,type=bind \
    --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
    --mount src=$TXT_DB,dst=/db,type=bind$RO \
    --mount src=$IMG_DIR,dst=/img,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -e NCCL_IB_CUDA_SUPPORT=0 \
    -w /src zmykevin/uc2:latest_version