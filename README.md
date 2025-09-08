# GemStoneClassfications

This repository implements a deep learning-based gemstone classification project using Convolutional Neural Network (CNN), VGG-19, and Xception models. Built in a Jupyter notebook (`GEM_stones_final.ipynb`) on Google Colab, it classifies gemstone images across 87 classes to tackle misidentification in the gemstone industry. The Xception model achieved the highest accuracy of 85.21%.

## Project Overview

With growing demand for gemstones in jewelry, driven by social media and ethical sourcing, accurate identification is crucial to avoid financial and reputational losses. This project leverages deep learning for robust classification, including data preprocessing (resizing to 255x255, augmentation), model evaluation (confusion matrix, classification report, accuracy/loss plots), and prediction visualization. Note that the notebook requires minor fixes (e.g., model definitions) for full reproducibility.

### Objectives
- Explore deep learning models for gemstone classification.
- Develop high-accuracy models for identification.
- Evaluate preprocessing techniques (resizing, augmentation).
- Assess performance using confusion matrix and classification report.
- Visualize predictions on sample images.

## Dataset

The dataset includes 9,063 resized gemstone images (255x255 pixels) from `/content/drive/MyDrive/data_sets/gemstones_resized_255`, covering 87 classes:
- **Total Images**: 9,063
- **Train-Test Split**: 80% training (7,250 images), 20% testing (1,813 images)
- **Data Loading**: Images are loaded into a Pandas DataFrame, shuffled, and split using `sklearn.model_selection.train_test_split`.
- **Source**: Derived from Kaggle's [Gemstone Images](https://www.kaggle.com/datasets/lsind18/gemstones-images), augmented to ~8,700 images (originally 3,200).

Download the dataset from Kaggle and resize images to 255x255.

## Models

Three models are evaluated:

1. **Custom CNN**:
   - Architecture: Convolutional layers (15 and 64 filters), max-pooling, dense layers (512, 256, 16 units).
   - Accuracy: 75.12%

2. **VGG-19**:
   - Pre-trained 19-layer model with 16 convolutional layers and 3 fully connected layers.
   - Accuracy: 79.59%

3. **Xception**:
   - Pre-trained model with depthwise separable convolutions (Entry, Middle, Exit Flows).
   - Accuracy: 85.21% (best performer)

The notebook includes evaluation code (confusion matrix, classification report) and visualizations (accuracy/loss plots, sample predictions).

## Hardware Requirements

- **Recommended**: Google Colab with GPU (T4 or better) for training.
- **Minimum**: 16 GB RAM, multi-core CPU.
- **Storage**: ~1 GB for dataset and model weights.
- **Environment**: Python 3.8+, TensorFlow/Keras, scikit-learn, Pandas, NumPy, Matplotlib.

## Installation

Install dependencies in Google Colab or locally:

```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python

## Installation
 Prerequisites
- Python: Version 3.8 or higher.
- Google Colab: Recommended for GPU support and Google Drive integration.
- Kaggle API: Optional, for downloading the dataset programmatically (requires a Kaggle account and API token).
- Git: For cloning the repository.
- Google Drive: For storing and accessing the dataset in Colab.
- Dependencies: Python packages listed in `requirements.txt`.

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<TummalapentaTejtarun>/GemStoneClassfication.git
   cd GemStoneClassfication


##Project Structure
textGemStoneClassfication/
├── GEM_stones_final.ipynb    # Jupyter notebook
├── data/                     # Dataset folder (gemstones_resized_255)
├── models/                   # Model weights (add after training)
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── report.pdf                # Project report (optional)
##Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch: git checkout -b feature/<feature-name>
Commit changes: git commit -m 'Add feature'
Push to the branch: git push origin feature/<feature-name>
Open a pull request.
