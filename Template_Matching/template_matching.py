import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from functions import visualize_images, template_matching, get_images, remove_background, scale_and_rotate

# Path to the folder containing the template images
template_folder = './original_data/template'

# Path to the folder containing the test images
test_folder = './original_data'

all_folders = [f.path for f in os.scandir(test_folder) if f.is_dir()]

# Remove the template folder from the list
all_folders.remove(os.path.join(test_folder, 'template'))

print('Template folder:', template_folder)
print('Test folder:', test_folder)
print('Searching in the following folders:', all_folders)

template_images = glob.glob(os.path.join(template_folder, '*.png'))

test_images = []
for folder in all_folders:
    test_images.extend(get_images(folder))

# Print the number of template images and test images
print('Number of template images:', len(template_images))
print('Number of test images:', len(test_images))

# Initialize counters
total_attempts = 0
successful_matches = 0
anomalies_counter = 0

# Initialize a counter for the images
image_counter = 0

# Iterate over each test image
for test_image_path in test_images:
    # Load the test image
    test_image = cv2.imread(test_image_path)

    print('Test image:', test_image_path)

    # Flag to indicate if a match was found
    match_found = False

    # Iterate over each template image
    for template_image_path in template_images:
        
        # Load the template
        template = cv2.imread(template_image_path)
        template = remove_background(template)
        
        # Scale and rotate the template
        transformed_templates = scale_and_rotate(template)

        for transformed_template in transformed_templates:
            # Visualize the images
            # visualize_images(transformed_template, test_image)
            pass
        
        image_counter += 1

        # Perform the template matching
        match = template_matching(test_image, transformed_template)

        # If a match is found, update the flag and break the loop
        if match:
            print(f"Match found with template {template_image_path}")
            match_found = True
            successful_matches += 1
            break
        
    # If no match was found after checking all templates, increment the anomalies counter
    if not match_found:
        print(f"No match found for test image {test_image_path}")
        anomalies_counter += 1
        plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        plt.show()

    total_attempts += 1

    print(f"Current anomaly count: {anomalies_counter}")
    print(f"Current successful match count: {successful_matches}")
    print(f"Total attempts so far: {total_attempts}")
    
#cv2.imwrite(f'processed_template_{image_counter}.png', processed_template)
print(f"Final number of anomalies: {anomalies_counter}")
print(f"Final number of successful matches: {successful_matches}")
print(f"Total attempts: {total_attempts}")

# Plot matches and anomalies
plt.plot(successful_matches, label='Matches')
plt.plot(anomalies_counter, label='Anomalies')
plt.xlabel('Attempt number')
plt.ylabel('Count')
plt.legend()
plt.show()
