from ultralytics import YOLO
import numpy as np


def predict_yolo(model_path, file):
    # load YOLO model
    model = YOLO(model_path)

    # predict on an image
    results = model(file)
    probs = results[0].cpu().probs.data.numpy()
    pred =  np.argmax(probs)

    return probs, pred


