import os
import pandas as pd
from train_oben import train_yolo
from predict import predict_yolo
from sklearn.metrics import accuracy_score

# Train YOLO and get the path to the best weights
best_weights_path = train_yolo()

# Define YOLO results path
yolo_path = best_weights_path  # Ensure we use the best weights from the training
names_dict = {0: 'Anomalie', 1: 'Schraube'}

# Directory containing test images
test_dir = './test_oben'

# Lists to store results
true_labels = []
pred_labels = []
probs_list = []

# Iterate over the test directory and make predictions
for root, _, files in os.walk(test_dir):
    for file in files:
        if file.endswith('.JPG') or file.endswith('.jpg'):
            # Get the full file path
            file_path = os.path.join(root, file)
            
            # Get the true label from the directory name
            true_label = os.path.basename(os.path.dirname(file_path))
            
            # Get the predicted probabilities and label
            probs_yolo, pred_yolo = predict_yolo(best_weights_path, file_path)
            
            # Append results to lists
            true_labels.append(true_label)
            pred_labels.append(names_dict[pred_yolo])
            probs_list.append(probs_yolo)

# Calculate overall accuracy
accuracy = accuracy_score(true_labels, pred_labels)

# Calculate detailed accuracy for each prediction
detailed_accuracy = [(1 if true == pred else 0) for true, pred in zip(true_labels, pred_labels)]

# Create a DataFrame to save results
results_df = pd.DataFrame({
    'File': [os.path.join(root, file) for root, _, files in os.walk(test_dir) for file in files if file.endswith('.JPG') or file.endswith('.jpg')],
    'True Label': true_labels,
    'Predicted Label': pred_labels,
    'Probabilities': probs_list,
    'Detailed Accuracy': detailed_accuracy
})

# Save results to CSV in the results directory
results_dir = './results_seitlich'
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, 'prediction_results.csv')
results_df.to_csv(csv_path, index=False)

# Print overall accuracy
print(f'Overall Accuracy: {accuracy}')

# Print detailed accuracy for each file
for index, row in results_df.iterrows():
    print(f"File: {row['File']} - True Label: {row['True Label']} - Predicted Label: {row['Predicted Label']} - Accuracy: {'100%' if row['Detailed Accuracy'] == 1 else '0%'}")
