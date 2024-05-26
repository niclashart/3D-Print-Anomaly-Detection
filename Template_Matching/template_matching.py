import cv2
import os
import glob

# Function to perform template matching
def template_matching(image, template):
    # Convert the images to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    template_resized = cv2.resize(template_gray, (image_gray.shape[1], image_gray.shape[0]))


    # Perform the template matching
    res = cv2.matchTemplate(image_gray, template_resized, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8

    # Check if the template and the image match
    if cv2.minMaxLoc(res)[1] >= threshold:
        return True
    else:
        return False

# Path to the folder containing the template images
template_folder = './original_data/template'

# Path to the folder containing the test images
test_folder = './original_data/Schraube-Zylinder_oben/seitlich'

# Print the paths to the folders
print('Template folder:', template_folder)
print('Test folder:', test_folder)

# Get the list of template images and test images
template_images = glob.glob(os.path.join(template_folder, '*.JPG'))
test_images = glob.glob(os.path.join(test_folder, '*.JPG'))

# Print the number of template images and test images
print('Number of template images:', len(template_images))
print('Number of test images:', len(test_images))

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

        # Perform template matching
        if template_matching(test_image, template_image):
            match_found = True
            break

    # Print the result
    if match_found:
        print('Match found')
    else:
        print('Anomaly detected')