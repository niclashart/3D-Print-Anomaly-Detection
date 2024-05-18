import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# load data


def load_training_data(train_path, val_path):

    # load data for cnn
    train_images, train_labels = (load_images_from_folder(train_path))
    train_images = train_images / 255.0

    val_images, val_labels = (load_images_from_folder(val_path))
    val_images = val_images / 255.0

    return train_images, train_labels, val_images, val_labels

# load test data from a single folder

def load_test_data(test_path):

    # load data for cnn
    test_images, test_labels = (load_images_from_folder(test_path))
    test_images = test_images / 255.0

    return test_images, test_labels

# load images from folders


def load_images_from_folder(folder):
    images = []
    labels = []
    class_labels = {'Anomalie': 1, 'Standard-Schraube': 0}

    for class_folder in os.listdir(folder):
        class_label = class_labels[class_folder]
        class_path = os.path.join(folder, class_folder)

        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = load_img(img_path, target_size=(32, 32))
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(class_label)

    return np.array(images), np.array(labels)


def load_image_from_file(file):
    img_path = os.path.join(file)
    img = load_img(img_path, target_size=(32, 32))
    img_array = img_to_array(img)
    return np.array(img_array)

# Laden der Bilder und Umwandeln in Arrays
def load_images_from_directory(directory, size):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                if image_path.endswith('.JPG') or image_path.endswith('.png'):  
                    img = load_img(image_path, target_size=size)
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(label)
    return np.array(images), np.array(labels)
