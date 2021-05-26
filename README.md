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
Download the pre-processed text features and pretrained checkpoints with the following command:
```
wget https://mmaisharables.blob.core.windows.net/uc2/UC2_DATA.tar.gz

```
The image features for mscoco can be obtained from [UNITER](https://github.com/ChenRocks/UNITER) via this [code script](https://github.com/ChenRocks/UNITER/blob/master/scripts/download_itm.sh). As CC's image features are large and inconvient for direct downloading, please contact UNITER's author to obtain the image features if you are interested in pretraining.

## Launch the Docker Container for Experiments
Once the user set up the data and checkpoints properly, please  run the following command to launch a docker container and start the pretraining process.
```
source launch_container_pretrain.sh /PATH_TO_STORAGE/txt_db /PATH_TO_STORAGE/img_db /PATH_TO_STORAGE/finetune /PATH_TO_STORAG/pretrain
```
## Pretraining
(Inside the Docker Container)If the user wants to run pretraining, please use the  following  command:
```
horovodrun -np $N_GPU python pretrain.py  --config config/uc2_pretrain.json
```
## Downstream Task Finetuning
**Text-to-Image Retrieval**
To run the finetuning experiment for the text-to-image retrieval task, please use the following command:
```
horovodrun -np $N_GPU python itm.py --config config/uc2_mscoco_itm.json
```

## Citation
If you find this code useful for your research, please consider citing:
```
@InProceedings{zhou2021uc,
author = {Zhou, Mingyang and Zhou, Luowei and Wang, Shuohang and Cheng, Yu and Li, Linjie and Yu, Zhou and Liu, Jingjing},
title = {UC2: Universal Cross-lingual Cross-modal Vision-and-Language Pre-training},
booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2021)},
year = {2021},
month = {June},
abstract = {Vision-and-language pre-training has achieved impressive success in learning multimodal representations between vision and language. To generalize this success to non-English languages, we introduce UC2 , the first machine translation-augmented framework for cross-lingual cross-modal representation learning. To tackle the scarcity problem of multilingual captions for image datasets, we first augment existing English-only datasets with other languages via machine translation (MT). Then we extend the standard Masked Language Modeling and Image-Text Matching training objectives to multilingual setting, where alignment between different languages is captured through shared visual context (i.e., using image as pivot). To facilitate the learning of a joint embedding space of images and all languages of interest, we further propose two novel pre-training tasks, namely Masked Region-to-Token Modeling (MRTM) and Visual Translation Language Modeling (VTLM), leveraging MT-enhanced translated data. Evaluation on multilingual image-text retrieval and multilingual visual question answering benchmarks demonstrates that our proposed framework achieves new state of the art on diverse non-English benchmarks while maintaining comparable performance to monolingual pre-trained models on English tasks.},
url = {https://www.microsoft.com/en-us/research/publication/uc2-universal-cross-lingual-cross-modal-vision-and-language-pre-training/},
}

```

## Acknowledge
Our code is mainly based on [Linjie Li](https://www.microsoft.com/en-us/research/people/linjli/) and [Yen-Chun Chen](https://www.microsoft.com/en-us/research/people/yenche/)'s project [UNITER](https://github.com/ChenRocks/UNITER). We thank the author for opening source their code and providing helful discussion for code implementation. Portions of the code also uses resources from [transformers](https://github.com/huggingface/transformers).

## Liscense
MIT
