import pandas as pd
import numpy as np
from helper_function import load_training_data, load_test_data
from train import train_cnn, train_minicnn
from predict import predict_cnn
from plot_metrics import plot_loss, plot_accuracy, plot_confusion_matrix

# Training

# define epoch and batch_size to compare models
epoch = 20
batch_size = 1

# load data for cnn 
train_path = "./Anomalieerkennung_DNN/data/train"
val_path = "./Anomalieerkennung_DNN/data/val"
train_images, train_labels, val_images, val_labels = load_training_data(train_path=train_path, val_path=val_path)

# train cnn
cnn_model, cnn_history = train_cnn(epoch, batch_size, train_images, train_labels, val_images, val_labels)
# train mini cnn
mini_cnn_model, mini_cnn_history = train_minicnn(epoch, batch_size, train_images, train_labels, val_images, val_labels)


# Prediction
file = './Anomalieerkennung_DNN/data/val/Schraube-Zylinder_ganz/IMG_8125.JPG'   # File to predict

names_dict = {0: "Schraube-ohne_Schraubansatz", 1: 'Schraube-Zylinder_ganz',
                     2: 'Schraube-Zylinder_oben',3: "Schraube-Zylinder_unten",
                       4: "Standard-Schraube",5: "Zylinder"}  # class names for cnn

probs_cnn, pred_cnn = predict_cnn(cnn_model, file)

probs_minicnn, pred_minicnn = predict_cnn(mini_cnn_model, file)

print(f"Class Dictionary: {names_dict}")

print(f"Probabilities CNN: {probs_cnn}")
print(f"Prediction CNN: {names_dict[pred_cnn]}")

print(f"Probabilities MiniCNN: {probs_minicnn}")
print(f"Prediction MiniCNN: {names_dict[pred_minicnn]}")

# Prediction on Test Data

test_path = "./Anomalieerkennung_DNN/data/val"
test_images, test_labels = load_test_data(test_path)

# predict on test data
cnn_test_probs = cnn_model.predict(test_images)
cnn_test_pred = np.argmax(cnn_test_probs, axis=1)

mini_cnn_test_probs = mini_cnn_model.predict(test_images)
mini_cnn_test_pred = np.argmax(mini_cnn_test_probs, axis=1)

# print counfusion matrix for both cnns

plot_confusion_matrix(test_labels, cnn_test_pred, "./Anomalieerkennung_DNN/cnn_confusion_matrix.png")
plot_confusion_matrix(test_labels, mini_cnn_test_pred, "./Anomalieerkennung_DNN/mini_cnn_confusion_matrix.png")

# Plot Results

plot_loss(cnn_history, mini_cnn_history)
plot_accuracy(cnn_history, mini_cnn_history)
