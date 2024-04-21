import pandas as pd
from train import train_yolo
from predict import predict_yolo
from plot_metrics import plot_loss, plot_accuracy

# Training

# define epoch and batch_size
epoch = 20


# train yolo
train_yolo(epoch)


# Prediction
file = '/home/jakob/4. Semester/Anomalieerkennung_YOLO/data/val/Anomalie/IMG_8117.JPG'   # File to predict

yolo_path = './runs/classify/train/weights/last.pt'

names_dict = {0 : 'Anomalie', 1 : "Standard-Schraube"}


probs_yolo, pred_yolo = predict_yolo(yolo_path, file)

print(f"Class Dictionary: {names_dict}")

print(f"Probabilities YOLO: {probs_yolo}")
print(f"Prediction YOLO: {names_dict[pred_yolo]}")


# Plot Results

# Define YOLO results path
yolo_history = pd.read_csv('./runs/classify/train/results.csv')

plot_loss(yolo_history)
plot_accuracy(yolo_history)
