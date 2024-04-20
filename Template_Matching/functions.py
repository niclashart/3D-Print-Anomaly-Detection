import cv2
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from helper_functions import load_data

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
  print(f"Template matching result: {result}")  # Check the result of the template matching
  _, max_val, _, max_loc = cv2.minMaxLoc(result)
  threshold = 0.01  # Adjust threshold for low correlation

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