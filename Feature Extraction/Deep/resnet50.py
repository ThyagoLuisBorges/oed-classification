import torch
import torchvision.models as models
import pandas as pd
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Labels for images
image_paths = healthy_images + mild_images + moderate_images + severe_images
labels = ['healthy'] * len(healthy_images) + ['mild'] * len(mild_images) + ['moderate'] * len(moderate_images) + ['severe'] * len(severe_images)

# Load a pretrained model, removing the classification head
# model = models.resnet50(pretrained=True)
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove final classification layer
model.eval()

# Image transformations for model input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Extract features
features = []
file_names = []  # List to store file names
total_images = len(image_paths)  # Total number of images

for img_path in tqdm(image_paths, desc="Processing Images", unit="image"):
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor).flatten().numpy()  # Flatten to 1D vector
    features.append(output)
    file_names.append(img_path.split('/')[-1])  # Extract file name

# Create DataFrame with features, labels, and file names
df_resnet50 = pd.DataFrame(features)
df_resnet50['label'] = labels
df_resnet50['file_name'] = file_names  # Add file names as a column