import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# Labels for images
image_paths = healthy_images + mild_images + moderate_images + severe_images
labels = ['healthy'] * len(healthy_images) + ['mild'] * len(mild_images) + ['moderate'] * len(moderate_images) + ['severe'] * len(severe_images)

# Load a pre-trained Vision Transformer model, removing the classification head
model = models.vit_b_16(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Image transformations for ViT model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT expects 224x224 input size
    transforms.ToTensor(),          # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Function to extract features from a single image
def extract_features(image_path):
    image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
    input_tensor = transform(image).unsqueeze(0)  # Apply transforms and add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
    return output.squeeze(0).numpy()  # Convert tensor to numpy array (1D feature vector)

# Process all images and extract features
features = []
file_names = []  # List to store file names
for img_path in tqdm(image_paths, desc="Extracting features", unit="image"):
    features.append(extract_features(img_path))
    file_names.append(os.path.basename(img_path))  # Extract the file name from the path

# Create a DataFrame for features, labels, and file names
df = pd.DataFrame(features)
df['label'] = labels  # Add the labels
df['file_name'] = file_names  # Add the file names