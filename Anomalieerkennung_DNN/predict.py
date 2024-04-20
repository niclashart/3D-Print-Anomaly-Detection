import numpy as np
import tensorflow as tf
from helper_function import load_image_from_file

def predict_cnn(cnn_model, file):
    # load image
    test_image = (load_image_from_file(file)) / 255.0
    # add batch-dimension
    test_image = tf.expand_dims(test_image, axis=0)

    probs = cnn_model.predict(test_image)
    pred = np.argmax(probs)

    return probs, pred
