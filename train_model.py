import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from model import ResNetViT
from preprocessing import extract_features, create_image_paths_dataset

# Paths to datasets
scab_path = '/kaggle/input/plantvillage-dataset/color/Apple___Apple_scab/'
black_path = '/kaggle/input/plantvillage-dataset/color/Apple___Black_rot/'
rust_path = '/kaggle/input/plantvillage-dataset/color/Apple___Cedar_apple_rust/'
healthy_path = '/kaggle/input/plantvillage-dataset/color/Apple___healthy/'

paths = [scab_path, black_path, rust_path, healthy_path]
labels = ['scab', 'black_rot', 'cedar_rust', 'healthy']
apples = create_image_paths_dataset(paths, labels)

# Split the data into training and testing sets
image_paths = apples['image_path'].tolist()
labels = apples['label'].tolist()
X_train_paths, X_test_paths, y_train, y_test = train_test_split(image_paths, labels, test_size=0.2, random_state=42)

# Instantiate the model
model = ResNetViT()
model.eval()  # Set to evaluation mode

# Extract features from the training and testing sets separately
features_train = extract_features(model, X_train_paths)
features_test = extract_features(model, X_test_paths)

# Standardize the features
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# Apply PCA for feature selection
pca = PCA(n_components=50)  # Adjust the number of components as needed
features_train_reduced = pca.fit_transform(features_train_scaled)
features_test_reduced = pca.transform(features_test_scaled)

# Initialize logistic regression model with L2 regularization
logistic_regression = LogisticRegression(max_iter=1000, penalty='l2', C=0.1)

# Cross-validation
cv_scores = cross_val_score(logistic_regression, features_train_reduced, y_train, cv=5)
print(f"Cross-Validation Accuracy: {cv_scores.mean()} ± {cv_scores.std()}")

# Fit the logistic regression model
logistic_regression.fit(features_train_reduced, y_train)

# Make predictions on the test set
y_pred = logistic_regression.predict(features_test_reduced)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
