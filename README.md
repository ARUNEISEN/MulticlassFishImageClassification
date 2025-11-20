# MulticlassFishImageClassification
This project classifies fish images into multiple categories using deep learning. It includes building a CNN from scratch and applying transfer learning with pre-trained models to improve accuracy. Models are saved for future use, and a Streamlit app is developed to predict fish categories from uploaded images.

A deep learning project to classify 11 species of fish using CNN, VGG16, ResNet50, MobileNetV2, and InceptionV3.

## Dataset

data/
│── train/ (80%)
│── val/   (10%)
└── test/  (10%)

## Setup

pip install -r requirements.txt

requirements file contains:
        tensorflow
        numpy
        pandas
        opencv-python
        Pillow
        scikit-learn
        matplotlib
        seaborn
        tqdm
        jupyter
        ipykernel
        h5py
## Training All Models
        CNN
        VGG16
        ResNet50
        MobileNetV2
        InceptionV3
## Model Evaluation
    Metrics generated:
        * Accuracy
        * Precision / Recall / F1-score
        * Classification Report
## Model Comparision Summary

        | Model           | Accuracy   | Macro Precision | Macro Recall | Macro F1 |
        | --------------- | ---------- | --------------- | ------------ | -------- |
        | CNN             | 0.9247     | 0.85            | 0.84         | 0.84     |
        | VGG16           | 0.9699     | 0.88            | 0.88         | 0.88     |
        | ResNet50        | 0.3037     | 0.36            | 0.26         | 0.19     |
        | **MobileNetV2** | **0.9937** | **1.00**        | 0.93         | 0.94     |
        | InceptionV3     | 0.9878     | 0.99            | 0.92         | 0.93     |

## Streamlit Web App

    Functions:

        *    Upload fish image

        *   Predict species

        *    Show confidence scores

        *    Display model info