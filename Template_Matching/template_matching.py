import cv2
import os
import glob
import matplotlib.pyplot as plt

# Function to perform template matching
def template_matching(image_gray, template_resized):
    # Perform the template matching
    res = cv2.matchTemplate(image_gray, template_resized, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8

    # Check if the template and the image match
    if cv2.minMaxLoc(res)[1] >= threshold:
        return True
    else:
        return False

def get_images(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        if 'oben' in dirs:
            dirs.remove('oben')  # don't visit 'oben' directories
        for file in files:
            if file.endswith('.JPG'):
                images.append(os.path.join(root, file))
    return images

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

template_images = glob.glob(os.path.join(template_folder, '*.JPG'))

test_images = []
for folder in all_folders:
    test_images.extend(get_images(folder))

# Print the number of template images and test images
print('Number of template images:', len(template_images))
print('Number of test images:', len(test_images))

# Initialize counters
total_attempts = 0
successful_matches = 0
accuracy_list = []
matches_list = []
anomalies_list = []

# Iterate over each test image
for test_image_path in test_images:
    # Load the test image
    test_image = cv2.imread(test_image_path)
    
    print('Test image:', test_image_path)

    # Flag to indicate if a match was found
    match_found = False

    # Iterate over each template image
    for template_image_path in template_images:
        # Load the template image
        template_image = cv2.imread(template_image_path)

        # Print the template image path and format
        print('Template image:', template_image_path)

        # Perform the template matching
        match = template_matching(test_image, template_image)

        # Update counters
        total_attempts += 1
        if match:
            successful_matches += 1
            match_found = True
            break  # Break the loop as soon as a match is found

    # Print the result and update the lists
    if match_found:
        print('Match found. Total matches so far:', successful_matches)
        matches_list.append(successful_matches)
    else:
        print('Anomaly detected. Total anomalies so far:', total_attempts - successful_matches)
        anomalies_list.append(total_attempts - successful_matches)

# Plot matches and anomalies
plt.plot(matches_list, label='Matches')
plt.plot(anomalies_list, label='Anomalies')
plt.xlabel('Attempt number')
plt.ylabel('Count')
plt.legend()
plt.show()