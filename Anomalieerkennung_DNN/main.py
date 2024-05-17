import pandas as pd
import numpy as np
from helper_function import load_training_data, load_test_data
from train import train_cnn, train_minicnn
from predict import predict_cnn
from plot_metrics import plot_loss, plot_accuracy, plot_confusion_matrix

import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Pfad zu Ihren Bildern
image_directory = './Anomalieerkennung_DNN/data_dnn/seitlich'

# Bildgröße angeben
image_size = (224, 224)

# Anzahl der augmentierten Bilder pro Originalbild
num_augmented_images_per_original = 10

# Laden der Bilder und Umwandeln in Arrays
def load_images_from_directory(directory, size):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                if image_path.endswith('.JPG') or image_path.endswith('.png'):  # Passen Sie dies an Ihre Bildformate an
                    img = load_img(image_path, target_size=size)
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(label)
    return np.array(images), np.array(labels)

# Bilder laden
images, labels = load_images_from_directory(image_directory, image_size)

# Normalisieren der Bilddaten
images = images.astype('float32') / 255.0

# Umwandeln der Labels in numerische Form (falls nötig)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Daten teilen
# Zuerst 60% Training und 40% temporär
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42, stratify=labels)
# Dann die temporären Daten in 50% Validierung und 50% Test teilen
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Überprüfen der Aufteilung
print('Trainingsdaten:', X_train.shape, y_train.shape)
print('Validierungsdaten:', X_val.shape, y_val.shape)
print('Testdaten:', X_test.shape, y_test.shape)

# Data Augmentation nur auf den Trainingsdaten
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Augmentierte Daten erzeugen und in Arrays speichern
augmented_images = []
augmented_labels = []

for i in range(len(X_train)):
    img = X_train[i]
    label = y_train[i]
    img = np.expand_dims(img, axis=0)
    j = 0
    for batch in train_datagen.flow(img, batch_size=1):
        augmented_images.append(batch[0])
        augmented_labels.append(label)
        j += 1
        if j >= num_augmented_images_per_original:
            break

# Konvertieren in numpy arrays
augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Überprüfen der Anzahl der erzeugten augmentierten Bilder
print('Augmentierte Trainingsdaten:', augmented_images.shape, augmented_labels.shape)

# Zusammenführen der ursprünglichen und augmentierten Trainingsdaten
X_train_augmented = np.concatenate((X_train, augmented_images))
y_train_augmented = np.concatenate((y_train, augmented_labels))

# Überprüfen der neuen Trainingsdaten
print('Neue Trainingsdaten:', X_train_augmented.shape, y_train_augmented.shape)

epoch = 30
batch_size = 2

# train cnn
cnn_model, cnn_history = train_cnn(epoch, batch_size, X_train_augmented, y_train_augmented, X_val, y_val)
# train mini cnn
mini_cnn_model, mini_cnn_history = train_minicnn(epoch, batch_size, X_train_augmented, y_train_augmented, X_val, y_val)


# Prediction on specific file
file_prediction = False

if file_prediction:
    file = './Anomalieerkennung_DNN/data/val/Schraube-Zylinder_ganz/IMG_8125.JPG'   # File to predict

    names_dict = {0: "Standard-Schraube", 1: 'Anomalie'}


    probs_cnn, pred_cnn = predict_cnn(cnn_model, file)

    probs_minicnn, pred_minicnn = predict_cnn(mini_cnn_model, file)

    print(f"Class Dictionary: {names_dict}")

    print(f"Probabilities CNN: {probs_cnn}")
    print(f"Prediction CNN: {names_dict[pred_cnn]}")

    print(f"Probabilities MiniCNN: {probs_minicnn}")
    print(f"Prediction MiniCNN: {names_dict[pred_minicnn]}")

# Prediction on Test Data

# predict on test data
cnn_test_probs = cnn_model.predict(X_test)
cnn_test_pred = np.argmax(cnn_test_probs, axis=1)
cnn_test_pred = label_encoder.inverse_transform(cnn_test_pred)  # Rücktransformation der numerischen Labels in die ursprünglichen Labels


mini_cnn_test_probs = mini_cnn_model.predict(X_test)
mini_cnn_test_pred = np.argmax(mini_cnn_test_probs, axis=1)
mini_cnn_test_pred = label_encoder.inverse_transform(mini_cnn_test_pred)  # Rücktransformation der numerischen Labels in die ursprünglichen Labels

# print counfusion matrix for both cnns
y_test = label_encoder.inverse_transform(y_test)

plot_confusion_matrix(y_test, cnn_test_pred, save_path="./Anomalieerkennung_DNN/cnn_confusion_matrix.png")
plot_confusion_matrix(y_test, mini_cnn_test_pred, "./Anomalieerkennung_DNN/mini_cnn_confusion_matrix.png")

# Plot Results

plot_loss(cnn_history, mini_cnn_history, save_path="./Anomalieerkennung_DNN/loss.png")
plot_accuracy(cnn_history, mini_cnn_history, save_path="./Anomalieerkennung_DNN/accuracy.png")
