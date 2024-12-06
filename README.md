# Mobile Mushroom Classifier

## Overview
This computer vision project focuses on object detection. The primary goal is to create a high quality classifier for species of mushrooms in the Continental United Stated, which can run locally on a mobile device. 

I am concerned primarily with the quality of classification and the size of the model. Because top 1 accuracy is low for smaller models relative to state of the art image classification models ex: [NoisyViT-B (95.5%, 348M param)](https://paperswithcode.com/sota/image-classification-on-imagenet) vs [MobileNetV3 Large (75.2% top 1, 5.4M param)](https://paperswithcode.com/paper/searching-for-mobilenetv3), it seems wise to return top five results and additional information for users to use their own judgement in attempting to identify mushrooms. At first glance, the latter model might seem completely incompetent. Truthfully, I do want to do better but must ballance the concern of size and compute. Furthermore, pre-trained versions of the MobileNetV3 model (among others) are highly accessible and abundant literature exists on training. Because of the concerns of availability and ease of use, I am going to focus, for the time being, on MobileNetV3, ResNet50, and Xception with Tensorflow, pre-trained on the ImageNet dataset. 

To fine tune these models, I have made various attempts. Initially, I simply looked on HuggingFace for image datasets of mushrooms. Maybe there's something I couldn't find, but high quality data seemed hard to access. I trained on what seemed like the best I could find on HuggingFace - a dataset of 215 species of mushrooms, sorted into directories based on their common name consisting of roughly 3K images. On average, this only amounts to about 13 training examples per class. Intuitively, this seems insufficient for training a deep neural net like MobileNetV3, and the results bear out that suspicion. Consequently, I have explored curating a larger dataset from iNaturalist. I used their publicly available csv files representing their datasets [accessible on GitHub here](https://github.com/inaturalist/inaturalist-open-data/tree/main) to curate a more robust dataset. At the time of writing, this dataset consists of 15,270 images in 110 classes. Clearly, this provides far more examples per class for a deep neural network to learn from. More details are provided in the [datasets](#Datasets) section of this README.

## Setup

### Python

There exists a requirements.txt file for this repository. It is generated with pipreqs. A virtual environment is my recommendation. the command 'pip install .' can be used to install all of the requirements.

### Settings / Config

I used a simple python file to store the config vars and import all from it in subsequent python files.

### CUDA / CUDNN

If you have a CUDA capable GPU (HIGHLY recommended), follow the [CUDA installation instructions](https://docs.nvidia.com/cuda/cuda-quick-start-guide/) for your system. Also install cuDNN, [instructions here](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-860/install-guide/index.html).

## Datasets

### Hugging Face: direct

From Hugging Face, I did use a dataset that I found insufficient. Although I would like to site the source, I can not locate it at this time and removed it from my file system to conserve space.

### iNaturalist: curated

I used the following [iNaturalist database information](https://github.com/inaturalist/inaturalist-open-data/tree/main) to compile a more robust dataset to test training, based on the hypothesis that the previously mentioned Hugging Face dataset was insufficient. I will not belabor the details of how exactly I went about this, but reach out directly if you would like to discuss ideas. It is important to note, however, that there are only two taxonomic subcategories of the kingdom of fungi I focused on - Division [Basidiomycota](https://en.wikipedia.org/wiki/Basidiomycota) and Phylum [Ascomycota](https://en.wikipedia.org/wiki/Ascomycota)

It is critical in supervised machine learning that the labeled data is not only sufficient in quantity, but also in quality. I have not yet come up with an automated approach to using iNaturalist's open-source data to ensuring the quality of the images. Although I only used "research" quality observations, some of the photos associated with those observations have issues shown in the following examples. It was tedious and time-consuming to manually look through the photos and delete offending photos, but I am considering an approach to automate this process of pruning the data with decent classifier.

*A finger is obstructing the view of the mushroom to too substantial a degree*
![Finger obstruction!](/assets/photos/meh_photos/finger.jpg)

*This observer posted a bunch of photos of a person examining the mushroom*
![Finger obstruction!](/assets/photos/meh_photos/dataset_corruption.png)

*Although potentially VERY useful for other purposes, microscopy photos wouldn't help classify images taken on a phone*
![microscopy!](/assets/photos/meh_photos/microsopy.jpg)

*this was a common occurrence...*
![environment!](/assets/photos/meh_photos/too_many_scenery.png)

*It might be useful for someone in the field to understand the typical environment A mushroom calls home, but it is not useful for image classification*
![environment!](/assets/photos/meh_photos/scenery.jpg)

*Although we want photos taken at a variety of distances, this might be a bit TOO far...*
![too far!](/assets/photos/meh_photos/too_far.jpg)

## Training

I used tensorflow to modify the models and retrain them to classify the species of mushrooms of interest. 

### Running on a GPU

A lot of the time, memory can critically impact training. to check the memory consumption on your GPU, run the command 'nvidia-smi'. look for processes consuming a lot of memory (like training a deep neural network *wink wink*). These processes can remain in memory for the GPU. Fortunately, a very simple troubleshooting step is to simply kill the process. In linux, the command to kill the process by its process Id (listed in the nvidia-smi output) is 'kill -9 PID' where PID is the process id; for example:
Below is the return of an nvidia-smi command

![nvidia-smi return!](/assets/photos/smi_return.png)

Notice that the following process is consuming a lot of resources:
 0   N/A  N/A    147886      C   python3                                      2128MiB 

To kill it, simple run a command like:

![kill process!](/assets/photos/kill_cmd.png)

## Results

### On a dataset with 