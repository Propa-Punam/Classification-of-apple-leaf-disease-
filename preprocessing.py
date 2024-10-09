import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Define image preprocessing with data augmentation
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(30),      # Randomly rotate images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features
def extract_features(model, image_paths):
    features = []
    for img_path in tqdm(image_paths):
        img = Image.open(img_path).convert('RGB')
        img = preprocess(img)
        img = img.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            feature = model(img)
        
        features.append(feature.squeeze().numpy())
    
    return np.array(features)

# Create dataset from image paths and labels
def create_image_paths_dataset(paths: list, labels: list):
    image_paths = []
    image_labels = []
    for path, label in zip(paths, labels):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        image_paths.extend([os.path.join(path, f) for f in files])
        image_labels.extend([label] * len(files))
    
    return pd.DataFrame({'image_path': image_paths, 'label': image_labels})
