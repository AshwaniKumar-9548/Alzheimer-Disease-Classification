# Alzheimer's Disease Classification Using Deep Learning

A comparative deep learning project for four-stage Alzheimer's disease classification from brain MRI images using **Vision Transformer (ViT)**, **Swin Transformer**, and **EfficientNet-B0**.

## Project Overview

This project implements and compares three deep learning architectures for multi-class Alzheimer's disease image classification.

The complete pipelines include:

- Image preprocessing and normalization
- Data augmentation
- Duplicate-image detection using file hashes
- Dataset auditing
- Train/validation splitting
- Class balancing and weighted sampling
- Transfer learning
- Model fine-tuning
- Early stopping
- Learning-rate scheduling
- Image-level evaluation
- Patient-level aggregation
- Classification reports
- Confusion matrices
- ROC curves
- Precision-recall curves
- Per-class precision, recall, and F1 analysis

## Models

### Vision Transformer (ViT)
A Transformer-based image classification architecture using global attention to learn relationships between image regions.

### Swin Transformer
A hierarchical Transformer architecture based on shifted-window attention.

### EfficientNet-B0
A convolutional neural network architecture designed for strong accuracy with computational efficiency.

## Repository Structure

```text
Alzheimer-Disease-Classification/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── ViT_Alzheimer.ipynb
│   ├── Swin_Alzheimer.ipynb
│   └── EfficientNet_B0_Alzheimer.ipynb
│
├── src/
│   ├── vit_alzheimer_pipeline.py
│   ├── swin_alzheimer_pipeline.py
│   └── efficientnet_b0_alzheimer_pipeline.py
│
├── results/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   ├── training_curves/
│   └── classification_reports/
│
├── models/
│   └── README.md
│
└── Dataset/
    └── README.md
```

## Dataset

The original MRI image dataset is **not included in this GitHub repository**.

The dataset should be obtained from its authorized original source and placed locally using the structure expected by the pipelines:

```text
Dataset/
├── Training Set/
│   ├── <Class 1>/
│   ├── <Class 2>/
│   ├── <Class 3>/
│   └── <Class 4>/
│
└── Testing Set/
    ├── <Class 1>/
    ├── <Class 2>/
    ├── <Class 3>/
    └── <Class 4>/
```

The exact class names must match between the training and testing directories.

## Installation

Clone the repository:

```bash
git clone https://github.com/<YOUR-USERNAME>/Alzheimer-Disease-Classification.git
cd Alzheimer-Disease-Classification
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipelines

After placing the dataset in the `Dataset/` directory:

### Vision Transformer

```bash
python src/vit_alzheimer_pipeline.py
```

### Swin Transformer

```bash
python src/swin_alzheimer_pipeline.py
```

### EfficientNet-B0

```bash
python src/efficientnet_b0_alzheimer_pipeline.py
```

The notebooks in `notebooks/` can also be opened and executed in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.

## Evaluation

The project evaluates models using:

- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted F1-score
- ROC-AUC
- Average Precision
- Confusion Matrix
- Classification Report

Both image-level and patient-level results are supported where patient grouping can be inferred from image filenames.

## Experimental Comparison

The purpose of the project is to compare ViT, Swin Transformer, and EfficientNet-B0 under a common four-class Alzheimer's classification task.

Place final figures, tables, and reports generated from your experiments in the `results/` directory.

## Reproducibility

The training pipelines use a fixed random seed and expose important training settings through configuration classes.

The code also performs dataset auditing, including duplicate detection and train/test exact-hash overlap checks.

## Important Note

This repository is intended for **academic and research purposes only**.

It is **not a medical diagnostic system** and must not be used for clinical diagnosis, treatment decisions, or patient care.

## Author

**Ashwani Kumar**

MCA – Data Science  
Bennett University

## Project Focus

Deep Learning | Computer Vision | Medical Image Classification | Transfer Learning | Vision Transformers | Swin Transformer | EfficientNet
