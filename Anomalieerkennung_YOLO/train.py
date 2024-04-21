from ultralytics import YOLO


def train_yolo(epoch):
    # load a pretrained model
    yolo_model = YOLO('yolov8n-cls.pt')
    # retrain yolo_model on data
    yolo_model.train(data='data', epochs=epoch, imgsz=64)

