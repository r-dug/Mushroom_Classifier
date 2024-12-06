# Mobile Mushroom Classifier

## Overview
This computer vision project focuses on object detection. The primary goal is to create a high quality classifier for species of mushrooms in the Continental United Stated, which can run locally on a mobile device. 

I am concerned primarily with the quality of classification and the size of the model. Because top 1 accuracy is low for smaller models relative to state of the art image classification models ex: [NoisyViT-B (95.5%, 348M param)](https://paperswithcode.com/sota/image-classification-on-imagenet) vs [MobileNetV3 Large (75.2% top 1, 5.4M param)](https://paperswithcode.com/paper/searching-for-mobilenetv3), it seems wise to return top five results and additional information for users to use their own judgement in attempting to identify mushrooms. At first glance, the latter model might seem completely incompetent. Truthfully, I do want to do better but must ballance the concern of size and compute. Furthermore, pre-trained versions of the MobileNetV3 model (among others) are highly accessible and abundant literature exists on training. Because of the concerns of availability and ease of use, I am going to focus, for the time being, on MobileNetV3, ResNet50, and Xception with Tensorflow, pre-trained on the ImageNet dataset. 

To fine tune these models, I have made various attempts. Initially, I simply looked on HuggingFace for image datasets of mushrooms. Maybe there's something I couldn't find, but high quality data seemed hard to access. I trained on what seemed like the best I could find on HuggingFace - a dataset of 215 species of mushrooms, sorted into directories based on their common name consisting of roughly 3K images. On average, this only amounts to about 13 training examples per class. Intuitively, this seems insufficient for training a deep neural net like MobileNetV3, and the results bear out that suspicion. Consequently, I have explored curating a larger dataset from iNaturalist. I used their publicly available csv files representing their datasets [accessible on GitHub here](https://github.com/inaturalist/inaturalist-open-data/tree/main) to curate a more robust dataset. At the time of writing, this dataset consists of 15,270 images in 110 classes. Clearly, this provides far more examples per class for a deep neural network to learn from. More details are provided in the [datasets](#Datasets) section of this README.

## Setup

### Python

### Settings / Config

### CUDA / CUDNN


## Datasets

### Hugging Face: direct

### iNaturalist: curated

## Training

### Running on a GPU

A lot of the time, memory can critically impact training. to check the memory consumption on your GPU, run the command 'nvidia-smi'. look for processes consuming a lot of memory (like training a deep neural network *wink wink*). These processes can remain in memory for the GPU. Fortunately, a very simple troubleshooting step is to simply kill the process. In linux, the command to kill the process by its process Id (listed in the nvidia-smi output) is 'kill -9 PID' where PID is the process id; for example, 

## Results
