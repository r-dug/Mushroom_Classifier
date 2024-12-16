# Mobile Mushroom Classifier

## Overview
The primary goal of this project is to create an image classification model for species of mushrooms, which can run locally on a mobile device. However, much of what is contained in this repository would easily transfer to creating other image classifiers, such as notes on curating a dataset and training a model with Tensorflow.

I am concerned primarily with ballancing model accuracy and size. Because top 1 accuracy is low for smaller models relative to state of the art image classification models ex: [NoisyViT-B (95.5%, 348M param)](https://paperswithcode.com/sota/image-classification-on-imagenet) vs [MobileNetV3 Large (75.2% top 1, 5.4M param)](https://paperswithcode.com/paper/searching-for-mobilenetv3), it seems wise to return top five results and additional information for users to use their own judgement in attempting to identify mushrooms. It is also, of course, a given that any application of this model does not constitute advice for the consumption of wild mushrooms. That goes for anyone reading this README as well! 

At first glance, the MobileNetV3 model might seem completely inadequate - *'~75% accuracy?! Ew!'* you might say (or think to yourself). and fair enough... This is a top 1 benchmark for the model when trained on the ImageNet dataset. Truthfully, I do want to achieve better results and believe that slight adjustments in implementing this model (or a similar one) could achieve them in this narrow domain. At any rate, MobileNetV3 is actually impressive for its relatively modest size. Furthermore, pre-trained versions of MobileNetV3 (among others) are highly accessible and abundant literature exists on [transfer learning](https://www.ibm.com/topics/transfer-learning#:~:text=Transfer%20learning%20is%20a%20machine,improve%20generalization%20in%20another%20setting.) for specific datasets. Because of the concerns of availability and ease of use, I am going to focus, for the time being, on MobileNetV3, ResNet50, Xception, and ConvNeXtV2 with Tensorflow, pre-trained on the ImageNet dataset (a de facto standard from what I can tell). 

To fine tune these models, one must find a suitable dataset or create one. First, I simply looked on [HuggingFace](https://huggingface.co/datasets?task_categories=task_categories:image-classification&sort=trending) for image datasets of mushrooms. I was unable to find a dataset I found suitably robust. Training on what seemed like the best I could find on HuggingFace - a dataset of 215 species, with roughly 3K samples total. On average, this only amounts to about 13 training examples per class. Intuitively, this seems insufficient for supervised learning (even with transfer learning and heavy data augmentation). The results seem to bear out that suspicion. Consequently, I have explored curating a larger dataset using iNaturalist. **Please note that iNaturalist clearly prohibits the monitization of their data. See section 7 of [this document](https://www.inaturalist.org/pages/terms)**. I used their publicly available csv files representing their datasets [accessible on GitHub here](https://github.com/inaturalist/inaturalist-open-data/tree/main), which absolutely has enough available data. More details are provided in the [datasets](#Datasets) section of this README.

## Setup

### Python

There exists a requirements.txt file for this repository. It is generated with [pipreqs](https://pypi.org/project/pipreqs/0.1.6/). Configuring a [virtual environment](https://docs.python.org/3/library/venv.html) is my recommendation. 

The command following can be used to install all of the requirements:

    > pip install -r requirements.txt

### Settings / Config

I used a simple python file to store the config vars and import that in subsequent python files. An example is available in this repo, but it will need to be modified to suit your needs (eg. specific paths).

### CUDA / CUDNN

If you have a CUDA capable GPU, follow the [CUDA installation instructions](https://docs.nvidia.com/cuda/cuda-quick-start-guide/) for your system. Also install cuDNN, [instructions here](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-860/install-guide/index.html) and the requisite tool kits for your environment. Note that if you are using Windows, you will need to use WSL... last time I personally tried that, it didn't work but there is WSL specific documentation on NVIDIA's site. If you don't have the hardware, you may want to use cloud compute to train your model; training even a model of MobileNetV3's size on a CPU would be prohibitively time expensive.

#### Troubleshooting note

The hardware available can critically impact training speed. In some cases, memory limitations can prohibit the use, much less the training of larger models. Furthermore, even training a model of *manageable* size for your hardware could fail because of memory consumption. I noticed the following issue. If a process consuming GPU memory does not exit gracefully, the memory might not deallocate correctly. To check the memory consumption on your GPU, run the command 'nvidia-smi'. Look for processes consuming a lot of memory and kill them if it is safe and appropriate to do so (but be careful, of course). In linux, the command to kill the process by its process ID (listed in the nvidia-smi output) is 'kill -9 PID' where PID is the process id.

Below is the return of an nvidia-smi command:

![nvidia-smi return!](/assets/photos/smi_return.png)

    > nvidia-smi

Notice that the following process is consuming a lot of resources:
 0   N/A  N/A    147886      C   python3                                      2128MiB 

To kill it, simply run a command like:

![kill process!](/assets/photos/kill_cmd.png)

    > kill -9 <PID in question>


## Datasets

### Hugging Face: direct

From Hugging Face, I found a mushroom image dataset but it was insufficient. 

### iNaturalist: curated

I used the following [iNaturalist database information](https://github.com/inaturalist/inaturalist-open-data/tree/main) to compile a more robust dataset to test training, based on the hypothesis that the Hugging Face dataset was insufficient. I will not belabor the details of how exactly I went about this, but reach out directly if you would like to discuss ideas. It is important to note, however, that there are only two taxonomic subcategories of the kingdom of fungi I focused on - Division [Basidiomycota](https://en.wikipedia.org/wiki/Basidiomycota) and Phylum [Ascomycota](https://en.wikipedia.org/wiki/Ascomycota). The iNaturalist GitHub site details steps for structuring the csv files into a PostreSQL database. Queries were fairly easy to write to return a table of N unique photos of M species, totaling N*M tuples. I saved this table as a csv, from which I could use a simple python script to compile an image dataset of sufficient size (for now, 194,258 samples accross 445 image classes (i.e. species)).

It is critical in supervised machine learning that the labeled data is not only sufficient in quantity, but also in quality. Although I only used "research" quality observations, some of the photos associated with those observations have issues shown in the following examples. To address the noise, I used the [DBSCAN clustering algorithm](https://scikit-learn.org/dev/modules/generated/sklearn.cluster.DBSCAN.html) found here to iterate over all of the image class directories and remove "anomolous" photos.

![Finger obstruction!](/assets/photos/meh_photos/finger.jpg)

*A finger is obstructing the view of the mushroom to too substantial a degree.*


![Finger obstruction!](/assets/photos/meh_photos/dataset_corruption.png)

*This observer posted a bunch of photos of a person examining the mushroom.*


![microscopy!](/assets/photos/meh_photos/microsopy.jpg)

*Although potentially VERY useful for other purposes, microscopy photos wouldn't help classify images taken on a phone.*


![environment!](/assets/photos/meh_photos/too_many_scenery.png)

*The sort of variety in the dataset*


![environment!](/assets/photos/meh_photos/scenery.jpg)

*It might be useful for someone in the field to understand the typical environment A mushroom calls home, but it is not useful for image classification*


![too far!](/assets/photos/meh_photos/too_far.jpg)

*Although we want photos taken at a variety of distances, this might be a bit TOO far...*

  **NOTE:** After some experimentation, I found some parameters to use with DBSCAN that were effective for this data. Running the algorithm over all classes ended up removing about 20,000 images from the dataset.


## Training

I used tensorflow to modify the models and retrain them to classify the species of mushrooms of interest. 

### Data Preparation

Tensorflow provides convenient methods to perform data preparation. Essentially, these are neural network layers used to format the data suitably for the model - including appropriate dimensions and scale. The resizing step is actually built into the tensorflow methor for creating an image dataset from a directory. Also, many tensorflow applications, MobileNetV3 included, include a rescaling step by default. 

### Data Augmentation

Tensorflow also provides convenient methods for data augmentation. There are various methods of including augmentation in your data pipeline. In this example, sequential models for augmentation are created and mapped onto the training dataset.

#### Randomized Operations

Below are links to documentation for the tensorflow data augmentation layers I used or experimented with.

- [RandomFlip](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomFlip)

- [RandomRotation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomRotation)

- [RandomTranslation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomTranslation)

- [RandomZoom](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomZoom)

- [RandomContrast](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomContrast)

- [RandomBrightness](https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomBrightness)

### Callbacks

Callbacks are used in training to either preserve progress or halt training in the event of plateaus. Tensorflow has implementations for various callbacks. Below are the ones I used, linking the tensorflow documentation, and a description (from tensorflow) of their general purpose.

- [ReduceLROnPlateau](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau)
      - Reduce learning rate when a metric has stopped improving.


- [ModelCheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)
      - Callback to save the Keras model or model weights at some frequency.


- [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
      - Stop training when a monitored metric has stopped improving.


### Training Steps

Convolution blocks are used as feature detectors. When tuning an existing classification model, it can be useful to use the layers already trained to encode features and assign class labels to the new image classes. After that step, the models deeper layers can be trained on the same data to further refine feature detection and improve model performance.


## Results

### MobileNetV3

![!phase 1](/assets/photos/training_performance/mobileCelia_0_0_1_MobileNetV3_phase_1_.png)

- No overfitting. That's nice.

![!phase 2](/assets/photos/training_performance/mobileCelia_0_0_1_MobileNetV3_phase_2_.png)

- Validation accuracy seems to plateau before training accuracy and the model seems to be overfitting (not generalizing well)

![!phase 3](/assets/photos/training_performance/mobileCelia_0_0_1_MobileNetV3_phase_3_.png)

- Overfitting continues

### ResNet50_V2

![!class training](/assets/photos/training_performance/mobileCelia_0_0_1_ResNet50V2_class_labels_lr_1e-3.png)

![!last 2 layers](/assets/photos/training_performance/mobileCelia_0_0_1_ResNet50V2_last_2_layers_lr_1e-4.png)

### Xception


