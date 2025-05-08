import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import mahotas as mh
import sys

def load_image(image_path):
    # Load image and convert to grayscale
    img = Image.open(image_path).convert('L')
    return np.array(img)

def compute_scm(image, distances=[1], angles=[0], levels=256):
    # Compute Spatial Co-occurrence Matrix
    scm = graycomatrix(image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    return scm

def extract_scm_features(scm):
    features = {}
    # Compute texture properties
    features['contrast'] = graycoprops(scm, 'contrast').mean()
    features['dissimilarity'] = graycoprops(scm, 'dissimilarity').mean()
    features['homogeneity'] = graycoprops(scm, 'homogeneity').mean()
    features['energy'] = graycoprops(scm, 'energy').mean()
    features['correlation'] = graycoprops(scm, 'correlation').mean()
    features['asm'] = graycoprops(scm, 'ASM').mean()
    return features

def process_images(image_paths, category_name, total_images):
    data = []
    processed_images = 0
    for image_path in image_paths:
        image = load_image(image_path)
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, np.pi]
        scm = compute_scm(image, distances=distances, angles=angles)
        features = extract_scm_features(scm)
        features['label'] = category_name
        data.append(features)
        processed_images += 1
        progress = (processed_images / total_images) * 100
        sys.stdout.write(f"\rProcessing {category_name} images: {progress:.2f}% completed.")
        sys.stdout.flush()
    return pd.DataFrame(data)