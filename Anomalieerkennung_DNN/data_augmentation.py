import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import random

def perform_data_augmentation(X_train, y_train, num_augmented_images_per_original, path):

    # Data Augmentation nur auf den Trainingsdaten (hier Parameter anpassen)
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

    # Beispiel: Anzeigen von zufällig ausgewählten augmentierten Trainingsbildern mit ihren Labels
    plt.figure(figsize=(10, 10))
    for i, img_index in enumerate(random.sample(range(len(augmented_images)), 9)):
        plt.subplot(3, 3, i+1)
        plt.imshow(augmented_images[img_index])
        plt.axis('off')
    plt.savefig(path)

    # Ausgabe der augmentierten Trainingsdaten

    return X_train_augmented, y_train_augmented

