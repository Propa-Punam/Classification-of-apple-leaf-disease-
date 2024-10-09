# Mango Leaf Disease Classification

This project implements a machine learning model to classify mango leaf diseases using a hybrid ResNet-ViT (Vision Transformer) architecture.

## Project Structure

```
MANGO LEAF/
├── LICENSE.md
├── model.py
├── preprocessing.py
├── README.md
├── requirements.txt
└── train_model.py
```

## Description

This project aims to classify mango leaf diseases using a combination of ResNet and Vision Transformer (ViT) models. The hybrid architecture leverages the strengths of both convolutional neural networks and transformer-based models for improved feature extraction and classification performance.

## Features

- Hybrid ResNet-ViT model for feature extraction
- Data augmentation techniques for improved generalization
- PCA for feature selection
- Logistic Regression classifier with L2 regularization
- Cross-validation for model evaluation

## Requirements

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset and update the paths in `train_model.py`.
2. Run the training script:

```
python train_model.py
```

## Results

The model achieves high accuracy in classifying mango leaf diseases:

- Cross-Validation Accuracy: 0.9897 ± 0.0036
- Test Accuracy: 0.9984

Detailed classification report:

```
              precision    recall  f1-score   support

   black_rot       1.00      1.00      1.00       114
  cedar_rust       1.00      0.98      0.99        61
     healthy       1.00      1.00      1.00       301
        scab       0.99      1.00      1.00       159

    accuracy                           1.00       635
   macro avg       1.00      1.00      1.00       635
weighted avg       1.00      1.00      1.00       635
```

## License

This project is licensed under the Apache 2.0 License.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

