import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from helper_functions import load_data
from train import create_cnn_model
from functions import get_roi, load_preprocess_image


epoch = 100
batch_size = 1

#load data 
train_path = "./data/train"
val_path = "./data/val"
train_images, train_labels, val_images, val_labels = load_data(train_path=train_path, val_path=val_path)

# Prediction data
image = 'C:/Users/nicla/Desktop/3danomaly/Template_Matching/data/val/Schraube-Zylinder_ganz/IMG_8125.JPG'   # File to predict

dict = {0: "Schraube-ohne_Schraubansatz", 1: 'Schraube-Zylinder_ganz',
                     2: 'Schraube-Zylinder_oben',3: "Schraube-Zylinder_unten",
                       4: "Standard-Schraube",5: "Zylinder"}  # class names for cnn

# Load template data
template_path = "C:/Users/nicla/Desktop/3danomaly/Template_Matching/data/template/image.png"
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
template = cv2.convertScaleAbs(template, alpha=(255.0))

# Resize template
template = cv2.resize(template, (28, 28))

# Training 
model, history = create_cnn_model(epoch, batch_size, train_images, train_labels, val_images, val_labels)

print(f"Dictionary of classes: {dict}")

template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
template = cv2.convertScaleAbs(template, alpha=(255.0))
train_features = []

# Resize template
template = cv2.resize(template, (28, 28))

print(f"Number of training images: {len(train_images)}")  # Check if there are any training images

# Preprocess training images and extract ROIs
for image, label in zip(train_images, train_labels):
  if isinstance(image, str):
    image = load_preprocess_image(image)
    print(f"Loaded image: {image.shape if image is not None else None}")  # Check if image is loaded correctly
    rois = get_roi(image, template)
    print(f"Found {len(rois) if rois else 0} ROIs")  # Check if any ROIs are found

    if rois:
        for roi in rois:
            roi_image = image[roi[1]:roi[3], roi[0]:roi[2]]
            features = ...  # Extract features from the ROI
            print(f"Added features: {features}")  # Check if features are added correctly
            train_features.append(features)
            

    else:
        train_features.append(np.zeros((28, 28)))  # Append default value
        print("No ROIs found, appending default value")

print(f"Training features {train_features}")

# Convert lists to NumPy arrays
train_features = np.array(train_features)
print(f"train_images shape: {train_features.shape}, size: {train_features.size}")
train_labels = np.array(train_labels)
