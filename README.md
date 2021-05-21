## UC2
[UC2: Universal Cross-lingual Cross-modal Vision-and-Language Pre-training](https://arxiv.org/abs/2104.00332)
<br/>
This  is the official repository of  UC2, a multili-lingual multi-modal pre-training framefork. In this repository we support end-to-end pretraining and finetuning for image-text retrieval on COCO. 

### Requirements
We Provide a Docker image to run our code. Please install the following:
- [nvidia driver (418+)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation),
- [Docker (19.03+)](https://docs.docker.com/engine/install/ubuntu/),
- [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

To run the docker command without sudo, user need to have [docker group membership](https://docs.docker.com/engine/install/linux-postinstall/). Our code only supports Linux with NVIDIA GPUs. We test our code on Ubuntu 18.04 and V100 cards.

## Data and Pretrained Checkpoints

Introduce how to download the processed data to be used for UC2.

## Pretraining
1. Once the user set up the data and checkpoints properly, please  run the following command to launch a docker container and start the pretraining process.
```
source launch_container_pretrain.sh /PATH_TO_STORAGE/txt_db /PATH_TO_STORAGE/img_db /PATH_TO_STORAGE/finetune /PATH_TO_STORAG/pretrained
```
2. If the user wants to run pretraining inside the container interactively, please use the  following  command:
```
horovodrun -np $N_GPU python pretrain.py  --config/uc2_pretrain.json
```
## Downstream Task Finetuning
**Text-to-Image Retrieval**
