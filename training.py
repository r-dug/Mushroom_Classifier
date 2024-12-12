import matplotlib.pyplot as plt
import numpy as np
import json

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
if BASE_MODEL == "Xception":
    from tensorflow.keras.applications import Xception
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Conv2D, Activation, Dropout, Flatten # layers to recreate input and output
elif BASE_MODEL == "ConvNeXtTiny":
    from tensorflow.keras.applications import ConvNeXtTiny
    from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Conv2D, Activation, Dropout, Flatten # layers to recreate input and output
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow as tf

# device check. train on gpu.
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {gpus} ")

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
plateau = ReduceLROnPlateau(    monitor="val_loss", 
                                mode="min", 
                                patience=5,
                                min_lr=1e-7, 
                                factor=0.5, 
                                min_delta=0.01,
                                verbose=1)

checkpointer = ModelCheckpoint( filepath=CHECKPOINT_PATH, 
                                verbose=1, 
                                save_best_only=True,
                                monitor="val_accuracy",
                                mode="max",
                                save_weights_only=True)

convergence = EarlyStopping(    monitor="val_accuracy",
                                min_delta=1e-4,
                                patience=7,
                                verbose=1,
                                mode="max",
                                baseline=None,
                                restore_best_weights=True,
                                start_from_epoch=10)

# datasets
train_data, val_data = image_dataset_from_directory(  DATASET_DIR,
                                            labels = 'inferred',
                                            label_mode = 'categorical',
                                            image_size=IMG_SIZE,
                                            validation_split = 0.25,
                                            subset = 'both',
                                            crop_to_aspect_ratio = True,
                                            seed=42
                                        )

class_names = train_data.class_names
n_classes = len(class_names)

try:
    with open('labels.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
except Exception as e:
    print(e)

if DEBUG_DATA == True:
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
if DEBUG_DATA == True:
    training_util.show_image_samples(train_data, class_names)
    training_util.show_image_samples(val_data, class_names)

train_data = training_util.configure_for_performance(train_data, AUTOTUNE=AUTOTUNE)
val_data = training_util.configure_for_performance(val_data, AUTOTUNE=AUTOTUNE)

inputs = Input(shape=(224,224, 3))

# establish a KerasTensor of layers for model output depending on base of choice in config 
if BASE_MODEL == "MobileNetV3":
    x, base = training_util.create_MobileV3(n_classes)
elif BASE_MODEL == "ResNet50V2":
    x, base = training_util.create_ResNet50V2(n_classes)
elif BASE_MODEL == "Xception":
    with_top = Xception( input_shape=INPUT_SHAPE, weights="imagenet", include_top=True)
    base = Xception( input_shape=INPUT_SHAPE, weights="imagenet", include_top=False)
    # x = training_util.create_Xception(n_classes)
elif BASE_MODEL == "ConvNeXtTiny":
    with_top = ConvNeXtTiny( input_shape=INPUT_SHAPE, weights="imagenet", include_top=True)
    base = ConvNeXtTiny( input_shape=INPUT_SHAPE, weights="imagenet", include_top=False)
    # x = training_util.create_ConvNeXtTiny(n_classes)

model = Model(inputs=base.input, outputs=x)
if DEBUG_MODEL == True:
    training_util.show_summary(base=base, with_top=None, model=model)

# let's just be sure to save the config to JSON for future investigation.
json_config = model.to_json()
with open(MODEL_JSON_PATH, 'w') as f:
    json.dump(json_config, f, indent=4)

# # optional to get weights from checkpoint file. commented out for fresh tune.
# try:
#     model.load_weights(CHECKPOINT_PATH)
#     print(f"loaded weights from: \n{CHECKPOINT_PATH}")
# except Exception as e:
#     print(e)

for layer in model.layers:
    layer.trainable = False


for layer in model.layers[PHASE_1_DEPTH:]:
    layer.trainable = True

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
    training_util.plot_performance(phase="phase_1_", 
                                   MODEL=MODEL_NAME,
                                   training_results=training_results)
    model.save(TF_MODEL_PATH)
except Exception as e:
   print(e)

# Phase 2 training
for layer in model.layers[PHASE_2_DEPTH:]:
    layer.trainable = True

model.compile(  optimizer=Adam(learning_rate=1e-3), 
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
    training_util.plot_performance(phase="phase_2_", 
                                   MODEL=MODEL_NAME,
                                   training_results=training_results)
    model.save(TF_MODEL_PATH)
except Exception as e:
   print(e)

#  phase 3 training
for layer in model.layers[PHASE_3_DEPTH:]:
    layer.trainable = True

# compile and with smaller learning rate and train again
model.compile(  optimizer=Adam(learning_rate=1e-4), 
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
    training_util.plot_performance(phase="phase_3_", 
                                   MODEL=MODEL_NAME,
                                   training_results=training_results)
    model.save(TF_MODEL_PATH)
except Exception as e:
   print(e)
