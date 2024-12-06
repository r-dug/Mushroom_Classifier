import os
import tensorflow as tf

VERSION = "0_1"
BASE_MODEL = "MobileNetV3" # ResNet50, MobileNetV3, or Xception in this project
MODEL_NAME = f"some_name_{VERSION}_{BASE_MODEL}"

# important dirs
DATASET_DIR = "/path/to/dataset/to/train/on"

TF_DIR = "/path/to/dataset/to/tensorflow/models/"
TFLITE_DIR = "/path/to/dataset/to/tensorflow_lite/models/"
CHECKPOINT_PATH = f"./checkpoints/{MODEL_NAME}.weights.h5"
TF_MODEL_PATH = os.path.join(TF_DIR, MODEL_NAME, ".keras")
TFLITE_MODEL_PATH = os.path.join(TFLITE_DIR, MODEL_NAME, ".tflite")

# data configuration
IMG_SIZE = (224, 224) # typical image size (in pixels) used in MobileNetV3, though others are acceptable
INPUT_SHAPE = (224,224,3) # typical input shape (pixels x pixels x channels) used in MobileNetV3, though others are acceptable
BATCH_SIZE = 16 # adjust batch size depending on your system
AUTOTUNE = tf.data.AUTOTUNE # this utility built in to tensorflow adjusts files loaded into memory