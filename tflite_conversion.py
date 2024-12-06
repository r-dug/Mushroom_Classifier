import tensorflow as tf
import numpy as np
from config import *
def generate_representative_data(X_data):
     print(X_data)
     num_samples = np.shape(X_data[0,0])[0]

     for i in range(num_samples):
          sample = X_data[i]
          yield {'input_1': sample}
def convert()->None:
    pre_model = tf.keras.models.load_model(f"{TF_MODEL_PATH}.keras")

    # Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(pre_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS 
    ]

    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    # Save the converted model
    with open(f"{TFLITE_MODEL_PATH}.tflite", "wb") as f:
        f.write(tflite_model)
        print("tflite model written")
if __name__ == "__main__":
     convert()