import pandas as pd
import numpy as np
from helper_function import load_images_from_directory
from data_augmentation import perform_data_augmentation
from train import train_cnn, train_minicnn
from predict import predict_cnn
from plot_metrics import plot_loss, plot_accuracy, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Pfad zu den Daten
image_directory = './Anomalieerkennung_DNN/data_dnn/seitlich'

# Parameter:

image_size = (224, 224)

use_augmentation = True

num_augmented_images_per_original = 50

epoch = 30

batch_size = 8

num_classes = len(os.listdir(image_directory))


# Bilder laden
images, labels = load_images_from_directory(image_directory, image_size)

# Normalisieren der Bilddaten
images = images.astype('float32') / 255.0

# Umwandeln der Labels in numerische Form (falls nötig)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Daten Aufteilen in Training, Validierung und Test
# Zuerst 58% Training und 42% temporär
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.42, stratify=labels)
# Dann die temporären Daten in 50% Validierung und 50% Test teilen
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

# Überprüfen der Aufteilung
print('Trainingsdaten:', X_train.shape, y_train.shape)
print('Validierungsdaten:', X_val.shape, y_val.shape)
print('Testdaten:', X_test.shape, y_test.shape)

if use_augmentation:
    X_train_augmented, y_train_augmented = perform_data_augmentation(X_train, y_train, num_augmented_images_per_original,
                                                                      path="./Anomalieerkennung_DNN/Auswertung/seitlich/augmented_images.png")
    X_train = X_train_augmented
    y_train = y_train_augmented


# CNN Trainieren
cnn_model, cnn_history = train_cnn(epoch, batch_size, image_size, num_classes, X_train, y_train, X_val, y_val)
# Mini-CNN Trainieren
mini_cnn_model, mini_cnn_history = train_minicnn(epoch, batch_size, image_size, num_classes, X_train, y_train, X_val, y_val)

# Vorhersage auf Testdaten
cnn_test_probs = cnn_model.predict(X_test)
cnn_test_pred = np.argmax(cnn_test_probs, axis=1)

mini_cnn_test_probs = mini_cnn_model.predict(X_test)
mini_cnn_test_pred = np.argmax(mini_cnn_test_probs, axis=1)

# Ausgabe Loss und Accuracy auf Testdaten

cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)
mini_cnn_test_loss, mini_cnn_test_accuracy = mini_cnn_model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss CNN: {cnn_test_loss}")
print(f"Test Accuracy CNN: {cnn_test_accuracy}")

print(f"Test Loss MiniCNN: {mini_cnn_test_loss}")
print(f"Test Accuracy MiniCNN: {mini_cnn_test_accuracy}")


cnn_test_pred = label_encoder.inverse_transform(cnn_test_pred)  # Rücktransformation der numerischen Labels in die ursprünglichen Labels
mini_cnn_test_pred = label_encoder.inverse_transform(mini_cnn_test_pred)  # Rücktransformation der numerischen Labels in die ursprünglichen Labels


# Ausgabe der Confusion Matrix für beide Modelle
y_test = label_encoder.inverse_transform(y_test)

plot_confusion_matrix(y_test, cnn_test_pred, save_path="./Anomalieerkennung_DNN/Auswertung/seitlich/cnn_confusion_matrix.png")
plot_confusion_matrix(y_test, mini_cnn_test_pred, "./Anomalieerkennung_DNN/Auswertung/seitlich/mini_cnn_confusion_matrix.png")

# Ergebnisse plotten

plot_loss(cnn_history, mini_cnn_history, save_path="./Anomalieerkennung_DNN/Auswertung/seitlich/loss.png")
plot_accuracy(cnn_history, mini_cnn_history, save_path="./Anomalieerkennung_DNN/Auswertung/seitlich/accuracy.png")

# Auswertung auf binäre Klassifikation (Anomalie vs. Normal) -> Standard-Schraube = 0; alle Fehlerbilder = 1

# Anomaliebilder als Klasse 1, Normale Bilder als Klasse 0
y_test_binary = np.where(y_test == "Standard-Schraube", 0, 1)
cnn_test_pred_binary = np.where(cnn_test_pred == "Standard-Schraube", 0, 1)
mini_cnn_test_pred_binary = np.where(mini_cnn_test_pred == "Standard-Schraube", 0, 1)

# Ausgabe der Genauigkeit

cnn_test_accuracy_binary = np.mean(y_test_binary == cnn_test_pred_binary)
mini_cnn_test_accuracy_binary = np.mean(y_test_binary == mini_cnn_test_pred_binary)

print(f"Test Accuracy CNN Binary: {cnn_test_accuracy_binary}")
print(f"Test Accuracy MiniCNN Binary: {mini_cnn_test_accuracy_binary}")
