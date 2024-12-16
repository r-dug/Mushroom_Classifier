import tensorflow.keras.layers as tfl
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import os
from config import *


# MobileNetV3, ResNet50V2, Xception, ConvNeXtTiny
if BASE_MODEL == "MobileNetV3":
    from tensorflow.keras.applications import MobileNetV3Large
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Conv2D, Activation, Dropout, Flatten # layers to recreate input and output

if BASE_MODEL == "ResNet50V2":
    from tensorflow.keras.applications import ResNet50V2
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Conv2D, Activation, Dropout, Flatten # layers to recreate input and output

if BASE_MODEL == "Xception":
    from tensorflow.keras.applications import Xception
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Conv2D, Activation, Dropout, Flatten # layers to recreate input and output

if BASE_MODEL == "ConvNeXtTiny":
    from tensorflow.keras.applications import ConvNeXtTiny
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Conv2D, Activation, Dropout, Flatten # layers to recreate input and output

# dataset configurations
def configure_for_performance(ds:tf.data.Dataset, AUTOTUNE)->None:
  '''
  a helper function to configure a dataset for better performance...
  current steps:
  - .cache() - stores a cache of the dataset. by default this method stores cache to memory, but our dataset os far to large so we store to disk.
  - .map() - uses our preprocessing layers, applied to the data on every epoch...
  - .shuffle() - shuffles the data, so training is less deterministic, and thus improved 
  - .prefetch() - CPU is utilized while GPU trains, to decrease the bottleneck of IO.

  These ops act on the dataset and aim to improve training efficiency.
  '''
  augmenter = data_augmenter()
  ds = (ds.prefetch(AUTOTUNE))
    # .shuffle(buffer_size=1000, reshuffle_each_iteration=True)
    # .map(augmenter, num_parallel_calls = AUTOTUNE)
    
  return ds

# rescaling (not relevant for all models, but rescales pixel vals from (0,255) to (-1,1) range)
def rescaler() ->tf.keras.Sequential:
    rescaler = tf.keras.Sequential([tf.keras.layers.Rescaling(scale=1./127.5)])
    return rescaler

# augmentation layers
def data_augmenter() -> tf.keras.Sequential:
    '''
    Create a Sequential model for augmenting image data. use before training 
    Returns:
        tf.keras.Sequential
    '''
    augmenter = tf.keras.Sequential([ 
    tfl.RandomFlip("horizontal and vertical"),
    tfl.RandomRotation(     factor = 0.2, 
                            fill_mode='nearest'),
    tfl.RandomTranslation(  height_factor = 0.05,
                            width_factor = 0.05,
                            fill_mode = "nearest",
                            interpolation = "nearest"),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.1, name="rand_contrast"),
    tf.keras.layers.RandomBrightness(0.1)
    ])
    
    return augmenter

def create_MobileV3(n):
    
    base = MobileNetV3Large(
        input_shape=INPUT_SHAPE,
        include_top = False,
        weights = "imagenet"    )

    # GlobalAveragePooling2D, Conv2D, Activation, Dropout, Flatten 
    x = base.output
    x = GlobalAveragePooling2D(name='final_pool_avg', trainable=True, data_format='channels_last', keepdims=True)(x)
    x = Conv2D( trainable=True,filters=1200, kernel_size=(1,1), strides=(1,1), padding='same', data_format='channels_last', dilation_rate= (1, 1), groups= 1, activation = 'linear', use_bias= True,)(x)
    x = Activation(name='activation_20', trainable = True, activation='hard_silu')(x)
    x = Dropout(rate=0.3)(x)
    x = Conv2D( trainable=True,filters=n, kernel_size=(1,1), strides=(1,1), padding='same', data_format='channels_last', dilation_rate= (1, 1), groups= 1, activation = 'linear', use_bias= True,)(x)
    x = Flatten(data_format='channels_last')(x)
    x = Activation(name='softmax_output',activation='softmax')(x)
    return x, base

def show_image_samples(ds:tf.data.Dataset, class_names:list[str]):
    plt.figure(figsize=(10, 10))
    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            normalized = images[i]/255 if tf.reduce_max(images) > 1.0 else images[i] 
            plt.imshow(normalized)
            plt.title(class_names[tf.argmax(labels[i]).numpy()])
            plt.axis("off")
    plt.show()

def show_summary(base, model, with_top):
    # compile and continue if so...
    #  check model sumaries for debugging if necessary
    print("\n\n***PRE-TRAINED MODEL:***\n\n")
    base.summary()

    print("\n\n***REVISED MODEL:***\n\n")
    model.summary()

def plot_performance(phase:str, training_results:tf.keras.callbacks.History) -> None:
    '''A simple logging function for the performance'''
    acc = [0.] + training_results.history['accuracy']
    val_acc = [0.] + training_results.history['val_accuracy']
    loss = training_results.history['loss']
    val_loss = training_results.history['val_loss']
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,max(max(val_loss), max(loss))])
    plt.title(f'Training and Validation Performance')
    plt.xlabel('epoch')
    plt.suptitle(phase, fontsize=16)
    plt.savefig(os.path.join(f"{RESULTS_DIR}",f"{MODEL_NAME}_{phase}.png"))
