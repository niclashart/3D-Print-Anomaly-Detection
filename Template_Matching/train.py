import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def create_cnn_model(epoch, batch_size, train_images, train_labels, val_images, val_labels):
  """
  Creates a simple CNN model for anomaly detection.
  """
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
  model.add(MaxPooling2D((2, 2)))    
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dense(6, activation='softmax'))

  # Compile CNN
  model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

  # fit CNN
  history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epoch, validation_data=(val_images, val_labels))

  history_df = pd.DataFrame(history.history)

  return model, history_df

