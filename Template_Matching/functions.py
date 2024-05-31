import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def visualize_images(template, test_image):
    # Create a new figure
    plt.figure()

    # Display the template
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    plt.title('Template')

    # Display the test image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title('Test Image')

    # Show the figure
    plt.show()
    plt.close()
  
    
# Function to perform template matching
def template_matching(test_image, template):
    # Convert the images to grayscale
    test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Set a threshold
    threshold = 0.51

    # Perform template matching in multiple scales
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # Resize the template according to the scale
        resized_template = cv2.resize(template_gray, (int(template_gray.shape[1] * scale), int(template_gray.shape[0] * scale)))

        # If the resized template is larger than the test image, break the loop
        if resized_template.shape[0] > test_image_gray.shape[0] or resized_template.shape[1] > test_image_gray.shape[1]:
            break

        # Perform template matching
        result = cv2.matchTemplate(test_image_gray, resized_template, cv2.TM_CCOEFF_NORMED)

        # Find the locations where the match score is above the threshold
        loc = np.where(result >= threshold)

        # If the number of matches is above 0, return True
        if len(loc[0]) > 0:
            return True

    # If no match was found, return False
    return False


def get_images(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.JPG'):
                full_path = os.path.join(root, file)
                if 'oben' not in full_path:
                    images.append(full_path)
    return images


def remove_background(image):
    mask = np.zeros(image.shape[:2], np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # Define the rectangle to include the whole image
    rect = (1, 1, image.shape[1]-2, image.shape[0]-2)

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

    image = image * mask2[:,:,np.newaxis]

    return image


def scale_and_rotate(image):
    # Define the scales to apply
    scales = [0.8, 1.0, 1.2]

    # Initialize a list to store the transformed images
    transformed_images = []

    # Iterate over each scale
    for scale in scales:
        # Resize the image
        resized_image = cv2.resize(image, None, fx=scale, fy=scale)

        # Iterate over each angle from 0 to 360
        for angle in range(360):
            # Get the image's dimensions
            (h, w) = resized_image.shape[:2]

            # Compute the center of the image
            center = (w / 2, h / 2)

            # Perform the rotation
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_image = cv2.warpAffine(resized_image, M, (w, h))

            # Add the transformed image to the list
            transformed_images.append(rotated_image)

    return transformed_images