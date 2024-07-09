import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_accuracy = 0
        self.epochs_no_improve = 0
        self.stop_training = False

    def on_epoch_end(self, current_accuracy):
        if current_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = current_accuracy
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            self.stop_training = True
            print("Early stopping triggered")

def get_latest_train_dir(base_dir):
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    train_dirs = [d for d in subdirs if d.startswith('train')]
    train_dirs.sort(reverse=True)
    if train_dirs:
        return os.path.join(base_dir, train_dirs[0])
    return None

def train_yolo():
    # Load a pretrained model
    yolo_model = YOLO('yolov8n-cls.pt')
    
    # Create early stopping object
    early_stopping = EarlyStopping()
    
    # Lists to store metrics
    val_accuracies = []
    
    # Train model with manual early stopping
    for e in range(50):
        results = yolo_model.train(data='./data_split_oben', epochs=1, imgsz=128)
        
        # Print the results dictionary keys to identify the correct keys
        print(f"Epoch {e+1} results: {results.results_dict}")
        
        # Extract metrics from results
        val_accuracy = results.results_dict.get('val_accuracy', results.top1)
        
        # Append metrics to lists
        if val_accuracy is not None:
            val_accuracies.append(val_accuracy)
        
        # Check for early stopping
        early_stopping.on_epoch_end(val_accuracy)
        if early_stopping.stop_training:
            print(f"Early stopping at epoch {e+1}")
            break

    # Determine the latest train directory and read the validation loss from CSV
    base_dir = '/home/user/3danomaly/Anomalieerkennung_YOLO/runs/classify'
    latest_train_dir = get_latest_train_dir(base_dir)
    
    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    # Save the accuracy plot to the results folder
    results_dir = './results_oben'
    os.makedirs(results_dir, exist_ok=True)
    accuracy_plot_path = os.path.join(results_dir, 'validation_accuracy.png')
    plt.savefig(accuracy_plot_path)

    # Return the path to the best weights
    best_weights_path = os.path.join(latest_train_dir, 'weights/best.pt')
    return best_weights_path

# Make sure this script doesn't run the train function unintentionally
if __name__ == "__main__":
    best_weights_path = train_yolo()
    print(f"Best weights saved at: {best_weights_path}")
