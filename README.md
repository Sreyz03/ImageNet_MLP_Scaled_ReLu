# ImageNet MLP Classification with Scaled ReLU

## Overview
This project implements image classification on a subset of the ImageNet dataset using a Multi-Layer Perceptron (MLP). The primary objective is to compare the performance of standard ReLU and roll-number-scaled ReLU activation functions under strict training constraints.

The model is trained using vanilla Stochastic Gradient Descent (SGD) with a fixed learning rate, and performance is evaluated using training loss, training accuracy, and validation accuracy.

This project is part of an academic deep learning assignment.

---

## Objectives
- Implement an MLP for multi-class image classification
- Train the network on an ImageNet subset
- Compare ReLU vs Scaled ReLU activation functions
- Analyze convergence behavior and accuracy trends
- Understand limitations of MLPs for image-based tasks

---

## Dataset
- Dataset: ImageNet (subset)
- Images are preprocessed by:
  - Resizing to a fixed resolution
  - Normalizing pixel values
  - Flattening images into 1-D vectors
- Data is split into training and validation sets

A subset is used due to the large size of the full ImageNet dataset.

---

## Model Architecture
The network is a fully connected MLP with the following structure:

| Layer | Description |
|------|------------|
| Input | Flattened image vector |
| Hidden Layer 1 | 64 neurons + ReLU / Scaled ReLU |
| Hidden Layer 2 | 128 neurons + ReLU / Scaled ReLU |
| Output Layer | C = 10 + r classes + Softmax |

Where:
- r = last two digits of the roll number (r = 62)
- Scaled ReLU is defined as:  
  f(x) = max(0, r · x)

---

## Training Configuration
- Optimizer: Vanilla SGD
- Learning rate: 0.01 (fixed)
- Epochs: 20
- Loss function: Cross-Entropy Loss
- Mini-batch training is used

Advanced optimizers and learning-rate schedulers are intentionally not used as per assignment constraints.

---

## Experiments
Two models are trained and compared:
1. MLP with Standard ReLU
2. MLP with Scaled ReLU

### Metrics
- Training loss vs epoch
- Training accuracy vs epoch
- Validation accuracy vs epoch

---

## Results and Observations
- Scaled ReLU shows faster initial convergence due to amplified gradients
- Standard ReLU provides more stable training
- Large scaling factors can introduce training instability
- Overall performance is limited by the absence of spatial feature extraction

---

## Limitations
- MLPs do not capture spatial relationships in images
- High parameter count for flattened image inputs
- Poor scalability to higher-resolution images
- Inferior performance compared to convolutional neural networks

---

## Repository Structure
imagenet-mlp-scaled-relu-roll62/
│
├── ImageNet_MLP_ScaledReLU_Roll62.ipynb
├── README.md
├── LICENSE
├── requirements.txt
├── results/
│ ├── training_loss.png
│ ├── train_accuracy.png
│ └── val_accuracy.png
└── weights/

---

## Requirements
- Python 3.8+
- PyTorch
- Torchvision
- NumPy
- Matplotlib

Install dependencies using:

---

## How to Run

1. Clone the repository:
git clone https://github.com/
<username>/imagenet-mlp-scaled-relu-roll62.git
cd imagenet-mlp-scaled-relu-roll62

2. Open and run the notebook:
   jupyter notebook ImageNet_MLP_ScaledReLU_Roll62.ipynb

---

## License
This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

## Author
Sreyas Kishore T  

