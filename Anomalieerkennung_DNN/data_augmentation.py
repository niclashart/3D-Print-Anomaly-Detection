import os
import torchvision.transforms as transforms
from PIL import Image

# Pfad zum Trainingsordner
train_folder = './data/train_seitlich/Anomalie'

# Pfad zum Ausgabeverzeichnis für die augmentierten Bilder
output_folder = './data/train_seitlich/Anomalie'

# Definieren Sie die gewünschten Transformationen für Data Augmentation
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)])

# Erstellen Sie das Ausgabeverzeichnis, falls es nicht existiert
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Durchsuchen Sie den Trainingsordner und wenden Sie Data Augmentation auf jedes Bild an
for filename in os.listdir(train_folder):
    if filename.endswith('.JPG'):
        # Öffnen Sie das Bild
        img_path = os.path.join(train_folder, filename)
        img = Image.open(img_path)

        # Wenden Sie Data Augmentation mehrmals an und speichern Sie die augmentierten Bilder (in diesem Fall 5-mal)
        for i in range(5):
            augmented_img = augmentation_transform(img)
            output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_{i}.jpg")
            augmented_img.save(output_path)

print("Data augmentation completed and augmented images saved to", output_folder)
