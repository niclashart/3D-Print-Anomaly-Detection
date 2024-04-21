import os
import torchvision.transforms as transforms
from PIL import Image

# Pfad zum Trainingsordner
train_folder = '/home/jakob/4. Semester/Anomalieerkennung_YOLO/data/train/Standard-Schraube'

# Pfad zum Ausgabeverzeichnis für die augmentierten Bilder
output_folder = '/home/jakob/4. Semester/Anomalieerkennung_YOLO/data/train/Standard-Schraube_augmented'

# Definieren Sie die gewünschten Transformationen für Data Augmentation
augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])

# Erstellen Sie das Ausgabeverzeichnis, falls es nicht existiert
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Durchsuchen Sie den Trainingsordner und wenden Sie Data Augmentation auf jedes Bild an
for filename in os.listdir(train_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Öffnen Sie das Bild
        img_path = os.path.join(train_folder, filename)
        img = Image.open(img_path)

        # Wenden Sie Data Augmentation an
        augmented_img = augmentation_transform(img)

        # Speichern Sie das augmentierte Bild im Ausgabeverzeichnis
        output_path = os.path.join(output_folder, filename)
        augmented_img.save(output_path)

print("Data augmentation completed and augmented images saved to", output_folder)
