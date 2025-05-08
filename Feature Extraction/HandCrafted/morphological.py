import glob
import cv2
import pandas as pd
import numpy as np


# Function to extract morphological features
def extract_morphological_features(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply binary thresholding
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = {}
    if contours:
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate area
        features['Area'] = cv2.contourArea(largest_contour)

        # Calculate perimeter
        features['Perimeter'] = cv2.arcLength(largest_contour, True)

        # Calculate circularity
        features['Circularity'] = (4 * np.pi * features['Area']) / (features['Perimeter'] ** 2) if features['Perimeter'] != 0 else 0

        # Calculate centroid
        M = cv2.moments(largest_contour)
        features['Centroid_X'] = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        features['Centroid_Y'] = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
    else:
        features = {'Area': 0, 'Perimeter': 0, 'Circularity': 0, 'Centroid_X': 0, 'Centroid_Y': 0}

    return features

# List to hold all feature data
data = []

# Process each category of images
for category, image_paths in zip(['healthy', 'mild', 'moderate', 'severe'],
                                  [glob.glob(segmented_images_healthy_path + '/*.png'),
                                   glob.glob(segmented_images_mild_path + '/*.png'),
                                   glob.glob(segmented_images_moderate_path + '/*.png'),
                                   glob.glob(segmented_images_severe_path + '/*.png')]):

    for image_path in image_paths:
        features = extract_morphological_features(image_path)
        features['label'] = category  # Add category label
        data.append(features)

# Create a DataFrame
features_df = pd.DataFrame(data)