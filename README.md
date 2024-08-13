# Classification-of-apple-leaf-disease-
This research aimed to develop a hybrid deep learning model leveraging both ResNet and Vision Transformer (ViT) architectures  for feature extraction

Data Preparation:

Dataset: The PlantVillage dataset was used, containing images of apple leaves categorized into four classes: Apple Scab, Black Rot, Cedar Apple Rust, and Healthy.
Image Preprocessing: A series of data augmentation techniques were applied, including resizing, random horizontal and vertical flips, random rotation, color jittering, and random grayscale conversion. Images were then normalized.
Hybrid Model Architecture:

ResNet50: A pre-trained ResNet50 model was used for its robust feature extraction capabilities. The final classification layer was removed to use the extracted features.
Vision Transformer (ViT): A pre-trained ViT model was integrated, which is effective in capturing long-range dependencies in images. Similar to ResNet50, the final classification layer was removed.
Dropout Layer: A dropout layer with a 50% dropout rate was added to prevent overfitting.
Custom Dataset Class:

A custom dataset class was implemented to handle the loading and preprocessing of images.
Feature Extraction:

Features were extracted from images using the hybrid model, combining the outputs from ResNet50 and ViT.
Data Split:

The dataset was split into training, validation, and test sets with a 70-15-15 ratio, ensuring stratification by class labels.
Feature Standardization and Dimensionality Reduction:

StandardScaler was used to standardize features. PCA was applied to reduce the dimensionality of the feature space.
Model Training:

Logistic Regression was chosen for the classification task. Class weights were computed to handle class imbalance.
Hyperparameter tuning was performed using Grid Search with cross-validation to optimize the logistic regression model.
Results
The hybrid ResNet-ViT model achieved significant performance metrics:

Validation Set:
Accuracy: 85.29%
Confusion Matrix:

[[ 85   1   4   3]
 [  2  35   1   3]
 [ 12   3 216  16]
 [ 10   8   7  70]]
Classification Report:

              precision    recall  f1-score   support
  black_rot       0.78      0.91      0.84        93
 cedar_rust       0.74      0.85      0.80        41
    healthy       0.95      0.87      0.91       247
       scab       0.76      0.74      0.75        95
  accuracy                           0.85       476
 macro avg       0.81      0.84      0.82       476
weighted avg 0.86 0.85 0.85 476

Accuracy: 87.61%
Confusion Matrix:

[[ 82   1   6   4]
 [  1  38   0   3]
 [  7   0 218  22]
 [  5   4   6  79]]
Classification Report:

              precision    recall  f1-score    support
  black_rot      0.863158  0.881720  0.872340   93.00000
 cedar_rust     0.883721  0.904762  0.894118   42.00000
    healthy    0.947826  0.882591  0.914046  247.00000
       scab      0.731481  0.840426  0.782178   94.00000
  accuracy     0.876050  0.876050  0.876050    0.87605
 macro avg     0.856547  0.877375  0.865671  476.00000
weighted avg 0.882904 0.876050 0.878098 476.00000

Conclusion
The hybrid ResNet-ViT model demonstrated effective classification of apple diseases with high accuracy and robust performance across both validation and test sets. The integration of ResNet50 and ViT architectures allowed for comprehensive feature extraction, capturing both local and global image features. This study highlights the potential of hybrid models in improving disease detection in agriculture, providing a reliable tool for farmers and researchers.
