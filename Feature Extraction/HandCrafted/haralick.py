import mahotas as mh
from mahotas.features import haralick
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix
import sys

# Function to compute the GLCM
def compute_glcm(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return glcm[:, :, 0, 0]

# Function to compute the Maximal Correlation Coefficient (MCC)
def compute_mcc(glcm):
    eigenvalues, _ = np.linalg.eig(glcm)
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
    if len(sorted_eigenvalues) > 1:
        return np.sqrt(np.abs(sorted_eigenvalues[1]))  # Second largest eigenvalue, ensure non-negative
    else:
        return 0  # If eigenvalues are not sufficient

# Function to extract Haralick features from an image path and add MCC
def extract_haralick_features(image_path):
    image = mh.imread(image_path, as_grey=True)
    image = (image * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8

    # Compute the Haralick features and take the mean across all angles
    features = mh.features.haralick(image, return_mean=True)

    # Compute the GLCM and then MCC
    glcm = compute_glcm(image)
    mcc = compute_mcc(glcm)

    # Append the MCC to the Haralick features
    features_with_mcc = np.append(features, mcc)

    return features_with_mcc


# Initialize a list to hold all features and corresponding labels
data = []

# Calculate the total number of images for progress tracking
total_images = len(healthy_images) + len(mild_images) + len(moderate_images) + len(severe_images)
processed_images = 0

# Combine healthy and severe images with their corresponding labels
image_groups = [(healthy_images, 'healthy'), (mild_images, 'mild'), (moderate_images, 'moderate'), (severe_images, 'severe')]

# Process images in one loop
for image_group, label in image_groups:
    for idx, img_path in enumerate(image_group, start=1):
        features = extract_haralick_features(img_path)
        file_name = img_path.split('/')[-1]  # Extract the file name from the path
        data.append(np.append(features, [label, file_name]))

        # Update progress
        processed_images += 1
        progress = (processed_images / total_images) * 100
        sys.stdout.write(f"\rProcessing {label} images: {progress:.2f}% completed.")
        sys.stdout.flush()

# Haralick feature names
haralick_feature_names = [
    "angular second moment",
    "contrast",
    "correlation",
    "variance",
    "inverse difference moment",
    "sum average",
    "sum variance",
    "sum entropy",
    "entropy",
    "difference variance",
    "difference entropy",
    "information measure of correlation 1",
    "information measure of correlation 2",
    "maximal correlation coefficient"
]

# Convert the list to a DataFrame for easier handling
columns = haralick_feature_names + ['label', 'file_name']
df_haralick = pd.DataFrame(data, columns=columns)