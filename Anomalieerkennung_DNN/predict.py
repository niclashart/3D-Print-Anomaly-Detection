import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def prepare_image(image_path, image_size=(224, 224)):
    """
    Lädt ein Bild, passt seine Größe an und konvertiert es in ein Array.
    
    :param image_path: Pfad zum Bild.
    :param image_size: Zielgröße des Bildes.
    :return: Vorbereitetes Bild als Array.
    """
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Bild zu einem Batch hinzufügen
    return img_array.astype('float32') / 255.0

def load_and_predict(model_path, image_path):
    """
    Lädt ein trainiertes Modell und macht eine Vorhersage für ein angegebenes Bild.
    
    :param model_path: Pfad zum trainierten Modell.
    :param image_path: Pfad zum Bild, für das eine Vorhersage gemacht werden soll.
    """
    # Modell laden
    model = load_model(model_path)
    
    # Bild vorbereiten
    prepared_image = prepare_image(image_path)
    
    # Vorhersage machen
    prediction = model.predict(prepared_image)
    
    return prediction
