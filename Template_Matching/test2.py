import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from helper_functions import load_data


epoch = 100
batch_size = 1

#load data 
train_path = "./data/train"
val_path = "./data/val"
train_images, train_labels, val_images, val_labels = load_data(train_path=train_path, val_path=val_path)



# Function for template matching and ROI extraction
def get_roi(image, template):
  """
  Performs template matching and extracts ROI based on low correlation regions.

  Args:
      image (np.ndarray): Grayscale test image.
      template (np.ndarray): Grayscale template image.

  Returns:
      list: List of ROIs (bounding boxes) as tuples (x_min, y_min, x_max, y_max).
  """

  result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
  _, max_val, _, max_loc = cv2.minMaxLoc(result)
  threshold = 0.9  # Adjust threshold for low correlation

  # Find all regions below the threshold (potential defects)
  rois = []
  h, w = template.shape
  for y in range(result.shape[0] - h + 1):
    for x in range(result.shape[1] - w + 1):
      if result[y, x] < threshold:
        rois.append((x, y, x + w, y + h))

  return rois

# Function to load and preprocess image
def load_preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image at {image_path}")

    # Convert image to the correct depth and type
    image = cv2.convertScaleAbs(image, alpha=(255.0))

    # Preprocess the image
    image = cv2.resize(image, (28, 28))  # Adjust image size based on your model architecture

    print(f"Image shape: {image.shape}")

    return image

# Load template and convert it to the correct depth and type
template_path = ('image.png')
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
template = cv2.convertScaleAbs(template, alpha=(255.0))

# Resize template
template = cv2.resize(template, (28, 28))

# Define CNN model (modify architecture based on data complexity)
def create_cnn_model():
  """
  Creates a simple CNN model for anomaly detection.
  """
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
  model.add(MaxPooling2D((2, 2)))    
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model
  


# Load template and training data (replace with your data loading logic)
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
train_images = "./train"
train_labels = [0]  # 0 for non-defective, 1 for defective
train_features = []

# Preprocess training images and extract ROIs
for image_path, label in zip(train_images, train_labels):
  image = load_preprocess_image(image_path)
  rois = get_roi(image, template)

  if rois:
    for roi in rois:
      roi_image = image[roi[1]:roi[3], roi[0]:roi[2]]
      features = ...  # Extract features from the ROI
      train_features.append(features)
      train_labels.append(label)  # Append label only when features are appended

  else:
    train_features.append(np.zeros((28, 28)))  # Append default value

# Convert lists to NumPy arrays
train_features = np.array(train_features)
print(f"train_images shape: {train_features.shape}, size: {train_features.size}")
train_labels = np.array(train_labels)

# Reshape image data for CNN (adjust based on your model)
train_features = train_features.reshape(-1, 28, 28, 1)

# Train the CNN model
cnn_model = create_cnn_model()
cnn_model.fit(train_features, train_labels, epochs=100)  # Adjust training parameters

ANOMALY_THRESHOLD = 0.4

def detect_anomaly(image_path):
  """
  Detects anomalies in an image using template matching and a CNN model.

  Args:
      image_path (str): Path to the test image.

  Returns:
      bool: True if a defect is detected, False otherwise.
  """

  # Load and preprocess image
  image = load_preprocess_image(image_path)

  # Perform template matching and extract ROIs
  rois = get_roi(image, template)
  print(f"Number of ROIs found: {len(rois)}")
  
  # Anomaly prediction using CNN
  is_anomaly = False
  for roi in rois:
    roi_image = image[roi[1]:roi[3], roi[0]:roi[2]]
    roi_image = roi_image.reshape(1, roi_image.shape[0], roi_image.shape[1], 1)  # Reshape for CNN

    # Predict anomaly score using the CNN model
    prediction = cnn_model.predict(roi_image)
    print(f"Prediction score for ROI: {prediction[0][0]}")  # Print prediction score
    if prediction[0][0] > ANOMALY_THRESHOLD:
        is_anomaly = True
        break  # Stop iterating through ROIs if anomaly is already detected

    cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
    cv2.imshow('ROIs', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

  return is_anomaly

# Example usage
test_image_path = "IMG_8167.JPG"
if detect_anomaly(test_image_path):
  print("Defect detected!")
else:
  print("Component seems OK.")
