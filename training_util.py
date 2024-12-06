import tensorflow.keras.layers as tfl
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from config import *

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

def show_summary(base, model):
    # compile and continue if so...
    #  check model sumaries for debugging if necessary
    print("\n\n***PRE-TRAINED MODEL:***\n\n")
    base.summary()
    print("\n\n***REVISED MODEL:***\n\n")
    model.summary()

def plot_performance(phase:str, training_results, MODEL:str) -> None:
    acc = [0.] + training_results.history['accuracy']
    val_acc = [0.] + training_results.history['val_accuracy']
    loss = training_results.history['loss']
    val_loss = training_results.history['val_loss']
    '''A simple logging function for the performance'''
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
    plt.ylim([0,2])
    plt.title(f'Training and Validation Performance')
    plt.xlabel('epoch')
    plt.savefig(f"./training_performance/{MODEL}_{phase}.png")

# dataset configurations
def configure_for_performance(ds:tf.data.Dataset, BATCH_SIZE:tuple[int,int], AUTOTUNE)->None:
  '''
  a helper function to configure a dataset for better performance...
  '''
  ds = ds.prefetch(buffer_size=AUTOTUNE).shuffle(buffer_size=1000).cache()
  return ds

def rescaler() ->tf.keras.Sequential:
    rescaler = tf.keras.Sequential([tf.keras.layers.Rescaling(scale=1./127.5)])
    return rescaler
# augmentation layer
def data_augmenter() -> tf.keras.Sequential:
    '''
    Create a Sequential model for augmenting image data. use before training 
    Returns:
        tf.keras.Sequential
    '''
    augmenter = tf.keras.Sequential([ 
    tfl.RandomFlip("horizontal and vertical"),
    tfl.RandomRotation(     factor = 0.3, 
                            fill_mode='nearest'),
    tfl.RandomTranslation(  height_factor = 0.1,
                            width_factor = 0.1,
                            fill_mode = "nearest",
                            interpolation = "nearest"),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1, name="rand_contrast"),
    tf.keras.layers.RandomBrightness(0.1)
    ])
    
    return augmenter
