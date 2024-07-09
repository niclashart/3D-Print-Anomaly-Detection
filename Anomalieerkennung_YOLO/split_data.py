import os
import shutil
from sklearn.model_selection import train_test_split

# Define the paths
data_dir = './data_oben'
train_dir = './data_split_oben/train'
val_dir = './data_split_oben/val'
test_dir = './test_oben'

# Create directories for train, val, test
for dir_path in [train_dir, val_dir, test_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Split ratio
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Ensure the ratios sum up to 1
assert train_ratio + val_ratio + test_ratio == 1.0, "The split ratios must sum to 1."

# Get the classes (subfolder names)
classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

    # Split the images
    train_images, temp_images = train_test_split(images, train_size=train_ratio, shuffle=True, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=test_ratio / (val_ratio + test_ratio), shuffle=True, random_state=42)

    # Copy images to respective directories
    for image_set, folder in [(train_images, train_dir), (val_images, val_dir), (test_images, test_dir)]:
        class_folder = os.path.join(folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        for image in image_set:
            shutil.copy(os.path.join(class_path, image), os.path.join(class_folder, image))

print("Dataset split completed.")
