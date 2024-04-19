import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models


def train_yolo(epoch):
    # load a pretrained model (recommended for training)
    # TODO: load pretrained yolov8n-cls model
    yolo_model = YOLO("yolov8n-cls.pt")

    # TODO: retrain yolo_model on weather_dataset (Hint: imgsz=64)
    yolo_model.train(data='weather_dataset', epochs=epoch, imgsz=64)


def train_cnn(epoch, batch_size, train_images, train_labels, val_images, val_labels):
    # define cnn
    cnn_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      padding='same', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax')])

    # compile cnn
    cnn_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

    # fit cnn
    history = cnn_model.fit(train_images, train_labels,
                            batch_size=batch_size, epochs=epoch, validation_data=(val_images, val_labels))

    cnn_history_df = pd.DataFrame(history.history)

    return cnn_model, cnn_history_df


def train_minicnn(epoch, batch_size, train_images, train_labels, val_images, val_labels):
    # define mini cnn
    mini_cnn_model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu',
                      padding='same', input_shape=(32, 32, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(3, activation='softmax')])

    # compile mini cnn
    mini_cnn_model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

    # fit mini cnn
    history = mini_cnn_model.fit(train_images, train_labels,
                                 batch_size=batch_size, epochs=epoch, validation_data=(val_images, val_labels))

    mini_cnn_history_df = pd.DataFrame(history.history)

    return mini_cnn_model, mini_cnn_history_df
