import pandas as pd
from helper_function import load_data
from train import train_cnn, train_minicnn
from predict import predict_cnn
from plot_metrics import plot_loss, plot_accuracy

# Training

# define epoch and batch_size to compare models
epoch = 10
batch_size = 1

# load data for cnn 
train_path = "./data"
val_path = "./data"
train_images, train_labels, val_images, val_labels = load_data(train_path=train_path, val_path=val_path)

# TODO: train cnn
cnn_model, cnn_history = train_cnn(epoch, batch_size, train_images, train_labels, val_images, val_labels)
# TODO: train mini cnn
mini_cnn_model, mini_cnn_history = train_minicnn(epoch, batch_size, train_images, train_labels, val_images, val_labels)


# Prediction

#file = 'shine100.jpg'   # File to predict

names_dict = {'Schraube-ohne_Schraubansatz': 0, 'Schraube-Zylinder_ganz': 1,
                     'Schraube-Zylinder_oben': 2, "Schraube-Zylinder_unten": 3,
                       "Standard-Schraube": 4, "Zylinder": 5}  # class names for cnn

#probs_cnn, pred_cnn = predict_cnn(cnn_model, file)

#probs_minicnn, pred_minicnn = predict_cnn(mini_cnn_model, file)

#print(f"Class Dictionary: {names_dict}")

#print(f"Probabilities CNN: {probs_cnn}")
#print(f"Prediction CNN: {names_dict[pred_cnn]}")

#print(f"Probabilities MiniCNN: {probs_minicnn}")
#print(f"Prediction MiniCNN: {names_dict[pred_minicnn]}")


# Plot Results

plot_loss(cnn_history, mini_cnn_history)
plot_accuracy(cnn_history, mini_cnn_history)
