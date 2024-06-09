from predict import load_and_predict
from sklearn.preprocessing import OneHotEncoder
import joblib

# Pfad zum Modell und zum Bild
model_path_seitlich = './Anomalieerkennung_DNN/Modell/seitlich/cnn_model.h5'
model_path_oben = './Anomalieerkennung_DNN/Modell/oben/cnn_model.h5'
image_path_seitlich = './original_data/Standard-Schraube/seitlich/IMG_8053.JPG'
image_path_oben = './original_data/Standard-Schraube/oben/IMG_8054.JPG'

# OneHotEncoder laden
one_hot_encoder_seitlich = joblib.load('./Anomalieerkennung_DNN/Modell/seitlich/one_hot_encoder.joblib')
one_hot_encoder_oben = joblib.load('./Anomalieerkennung_DNN/Modell/oben/one_hot_encoder.joblib')

# Vorhersage machen seitliche Ansicht
prediction_seitlich = load_and_predict(model_path_seitlich, image_path_seitlich)

# OneHotEncoder umkehren, um die Klasse zu erhalten
predicted_class_seitlich = one_hot_encoder_seitlich.inverse_transform(prediction_seitlich)
print("Vorhersage seitliche Ansicht:", prediction_seitlich)
print("Vorhersage Klasse seitliche Ansicht:", predicted_class_seitlich)

# Vorhersage machen obere Ansicht
prediction_oben = load_and_predict(model_path_oben, image_path_oben)

# OneHotEncoder umkehren, um die Klasse zu erhalten
predicted_class_oben = one_hot_encoder_oben.inverse_transform(prediction_oben)
print("Vorhersage obere Ansicht:", prediction_oben)
print("Vorhersage Klasse obere Ansicht:", predicted_class_oben)

# Zusammenfassung der Vorhersagen
print("Zusammenfassung:")
if predicted_class_seitlich =="Standard-Schraube" and predicted_class_oben =="Standard-Schraube":
    print("Die Schraube ist in Ordnung.")

else:
    print("Die Schraube ist defekt.")
