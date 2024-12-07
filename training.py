import matplotlib.pyplot as plt
import numpy as np

# DATES AND TIMES... DUH
from datetime import datetime
# local
import training_util
from config import *

# deep learning with tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, BatchNormalization, Dropout, Input
if BASE_MODEL == "ResNet50V2":
    from tensorflow.keras.applications import ResNet50V2
if BASE_MODEL == "MobileNetV3":
    from tensorflow.keras.applications import MobileNetV3Large
if BASE_MODEL == "Xception":
    from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf

# device check. train on gpu.
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {gpus} ")
# input("CONTINUE? [enter]")

# GPU config
try:
    tf.config.run_functions_eagerly(True)
    with tf.init_scope():
        print(tf.executing_eagerly())
    print(tf.executing_eagerly())
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
    tf.data.experimental.enable_debug_mode()
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except RuntimeError as e:
  # Virtual devices must be set before GPUs have been initialized
  print(e)

# callback configurations
plateau = ReduceLROnPlateau(    monitor="val_loss", mode="min", patience=5,
                                min_lr=1e-7, factor=0.3, min_delta=0.01,
                                verbose=1)

checkpointer = ModelCheckpoint( filepath=CHECKPOINT_PATH, 
                                verbose=1, save_best_only=True,
                                monitor="val_accuracy", mode="max",
                                save_weights_only=True)

convergence = EarlyStopping(    monitor="val_accuracy",
                                min_delta=0e-5,
                                patience=7,
                                verbose=1,
                                mode="max",
                                baseline=None,
                                restore_best_weights=True,
                                start_from_epoch=15)

# datasets
train_data = image_dataset_from_directory(DATASET_DIR,
                                            labels = 'inferred',
                                            label_mode = 'categorical',
                                            shuffle=True,
                                            image_size=IMG_SIZE,
                                            seed=4242,
                                            validation_split = 0.4,
                                            subset = 'training',
                                            crop_to_aspect_ratio = True, 
                                            batch_size=BATCH_SIZE)

val_data = image_dataset_from_directory(DATASET_DIR,
                                            labels = 'inferred',
                                            label_mode = 'categorical',
                                            shuffle=True,
                                            image_size=IMG_SIZE,
                                            seed=4242,
                                            validation_split = 0.4,
                                            subset = 'validation',
                                            crop_to_aspect_ratio = True, 
                                            batch_size=BATCH_SIZE)

class_names = train_data.class_names
try:
    with open('labels.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
except Exception as e:
    print(e)

n_classes = len(class_names)

if DEBUG == True:
    training_util.show_image_samples(train_data, class_names)
    training_util.show_image_samples(val_data, class_names)

rescale = training_util.rescaler()
augmenter = training_util.data_augmenter()

if BASE_MODEL == "ResNet50V2": 
    augmenter.add(rescale.layers[0])
    val_data = train_data.map(
        lambda x, y: (rescale(x, training=True), y),
    )

train_data = train_data.map(
    lambda x, y: (augmenter(x, training=True), y),
    )
if DEBUG == True:
    training_util.show_image_samples(train_data, class_names)
    training_util.show_image_samples(val_data, class_names)

train_data = training_util.configure_for_performance(train_data, BATCH_SIZE=BATCH_SIZE, AUTOTUNE=AUTOTUNE)
val_data = training_util.configure_for_performance(val_data, BATCH_SIZE=BATCH_SIZE, AUTOTUNE=AUTOTUNE)

# starting with the MobileNetV3, mofidy the layers 
if BASE_MODEL == "ResNet50V2":
    base = ResNet50V2( input_shape=INPUT_SHAPE, weights="imagenet", include_top=False)
elif BASE_MODEL == "MobileNetV3":
    base = MobileNetV3Large( input_shape=INPUT_SHAPE, weights="imagenet", include_top=False)
elif BASE_MODEL == "Xception":
    base = Xception( input_shape=INPUT_SHAPE, weights="imagenet", include_top=False)

base.trainable = False

inputs = Input(shape=(224,224, 3))
x = base(inputs, training=False)
x = GlobalMaxPooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dense(n_classes, activation='softmax')(x) # for new output
model = Model(inputs, x)

if DEBUG == True:
    training_util.show_summary(base, model)
# # optional to get weights from checkpoint file. commented out for fresh tune.
# model.load_weights(CHECKPOINT_PATH)

#  only the last layer trainable: fit categories
for layer in model.layers:
    layer.trainable = False
model.layers[-1].trainable = True

model.compile(  optimizer=Adam(learning_rate=1e-3), 
                loss=tf.keras.losses.CategoricalCrossentropy(), 
                metrics=['accuracy'])
training_results = model.fit(   train_data, 
                                epochs=50, 
                                verbose=1,
                                callbacks=[plateau, checkpointer, convergence],
                                validation_data=val_data,
                                batch_size=BATCH_SIZE
                                )
try:
    training_util.plot_performance(phase="class_labels_lr_1e-3", 
                                   MODEL=MODEL_NAME,
                                   training_results=training_results)
    model.save(TF_MODEL_PATH)
except Exception as e:
   print(e)

#  only the last 20 layers trainable
for layer in model.layers[-2:]:
    layer.trainable = True

# compile and with smaller learning rate and train again
model.compile(  optimizer=Adam(learning_rate=1e-4), 
                loss=tf.keras.losses.CategoricalCrossentropy(), 
                metrics=['accuracy'])
training_results = model.fit(   train_data, 
                                batch_size=BATCH_SIZE,
                                epochs=50, 
                                verbose=1,
                                callbacks=[plateau, checkpointer, convergence],
                                validation_data=val_data,
                                )

try:
    training_util.plot_performance(phase="last_2_layers_lr_1e-4", 
                                   MODEL=MODEL_NAME,
                                   training_results=training_results)
    model.save(TF_MODEL_PATH)
except Exception as e:
   print(e)

#  last 30 layers trainable
for layer in model.layers:
    layer.trainable = True

# compile and with smaller learning rate and train again
model.compile(  optimizer=Adam(learning_rate=1e-6), 
                loss=tf.keras.losses.CategoricalCrossentropy(), 
                metrics=['accuracy'])
training_results = model.fit(   train_data, 
                                epochs=100, 
                                verbose=1,
                                callbacks=[plateau, checkpointer, convergence],
                                validation_data=val_data,
                                batch_size=BATCH_SIZE,
                                )

try:
    training_util.plot_performance(phase="whole_network_lr_1e-6", 
                                   MODEL=MODEL_NAME,
                                   training_results=training_results)
    model.save(TF_MODEL_PATH)
except Exception as e:
   print(e)
