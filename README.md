# Skin Cancer Classification using Hybrid ResNet18

A deep learning project for multi-class skin cancer classification using PyTorch Lightning and a hybrid ResNet18 architecture.

## 📋 Project Overview

This project implements a computer vision system for classifying skin lesions into 9 different categories using transfer learning with ResNet18. The model achieves **84.14% accuracy** on the test set with balanced performance across multiple skin disease categories.

## 🏗️ Architecture

### Model Design
- **Backbone**: Pretrained ResNet18 with frozen early layers
- **Classification Head**: Custom sequential layers (512 → 128 → num_classes)
- **Hybrid Training**: Differential learning rates for backbone (3e-5) and classifier (8e-4)
- **Regularization**: Dropout (0.3) and weight decay

### Key Features
- **Transfer Learning**: Leverages ImageNet pretrained weights
- **Class Imbalance Handling**: WeightedRandomSampler for balanced training
- **Data Augmentation**: Rotation, flipping, color jittering, and normalization
- **Early Stopping**: Prevents overfitting with patience=5
- **Multi-format Export**: TorchScript, ONNX, and PyTorch formats

## 📊 Dataset
The dataset is available here:
[https://drive.google.com/drive/folders/1qyNY30YjTspWv2nZyNOss7RVceHcvpTE](https://drive.google.com/drive/folders/1qyNY30YjTspWv2nZyNOss7RVceHcvpTE?usp=sharing)

### Classes (9 Categories)
1. Actinic Keratosis
2. Basal Cell Carcinoma
3. Dermatofibroma
4. Melanoma
5. Nevus
6. Pigmented Benign Keratosis
7. Seborrheic Keratosis
8. Squamous Cell Carcinoma
9. Vascular Lesion

### Data Split
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%

## 🚀 Performance

### Test Results
- **Accuracy**: 85.92%
- **Loss**: 0.459
- **Macro F1-score**: 84.02%

### Per-Class Performance (F1-scores)
- Nevus: 93.57%
- Pigmented Benign Keratosis: 94.52%
- Vascular Lesion: 91.57%
- Melanoma: 87.36%
- Dermatofibroma: 82.17%
