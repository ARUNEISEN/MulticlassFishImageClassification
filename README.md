# Multiclass Fish Image Classification

## Project Overview

The **Multiclass Fish Image Classification** project focuses on classifying fish images into multiple categories using Deep Learning techniques.

### The project includes:

- Training a Custom CNN from scratch  
- Applying Transfer Learning with multiple pre-trained models  
- Model comparison & evaluation  
- Saving the best-performing model  
- Deploying a Streamlit web application for real-time predictions  

---

## Domain

Deep Learning (Computer Vision – Image Classification)

---

## Problem Statement

Accurately classify fish images into their respective species using deep learning models.

### Key Challenges:

- Handling multi-class classification  
- Improving generalization using augmentation  
- Reducing model bias  

---

## Business Use Cases

### 1️. Enhanced Accuracy  
Determine the best model architecture for fish classification.

### 2️. Model Comparison  
Evaluate multiple CNN architectures and select the most efficient one.

### 3️. Deployment Ready  
Provide a real-time prediction interface via a web application.

---

## Skills Gained

- Deep Learning  
- Python  
- TensorFlow / Keras  
- Transfer Learning  
- Data Preprocessing & Augmentation  
- Model Evaluation Metrics  
- Visualization  

---

## Project Structure


MULTICLASSFISHIMAGECLASSIFICATION/
│
├── data/
│   ├── train/
│   └── val/
│
├── Notebooks/
│   ├── Data_Analysis_and_Preprocessing.ipynb
│   ├── Data_Generators_and_FocalLoss.ipynb
│   ├── Model_Training.ipynb
│   ├── Model_Evaluation_and_Comparison.ipynb
│   └── models/
│       ├── CustomCNN_best.h5
│       ├── EfficientNetB0_best.h5
│       ├── InceptionV3_best.h5
│       ├── MobileNet_best.h5
│       ├── ResNet50_best.h5
│       └── model_metadata.json
│
├── results/
│   ├── confusion_matrices/
│   ├── metrics/
│   └── plots/
│
├── FishImageClassifier.py
├── requirements.txt
└── README.md
```

---

## Dataset

- Multi-class fish image dataset  
- Images organized into class-specific folders  
- Loaded using TensorFlow ImageDataGenerator  
- Provided as ZIP file  

---

## Project Workflow

### 1️. Data Preprocessing & Augmentation

- Rescaled images to [0,1]  
- Applied:
  - Rotation  
  - Zoom  
  - Horizontal Flip  

---

### 2️. Model Training

#### Custom CNN
- Convolution + Pooling + Dense layers  

#### Transfer Learning Models Used

- VGG16  
- ResNet50  
- MobileNet  
- InceptionV3  
- EfficientNetB0  

---

### 3️. Model Evaluation

Models were compared using:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  
- Training & Validation Curves  

### Visualizations Included:

- Accuracy vs Epoch  
- Loss vs Epoch  
- Confusion Matrices  

---

### 4️. Model Saving

Best-performing models saved in:

.h5 format

---

## Streamlit Deployment

An interactive web application was built using Streamlit.

### Features:

- Upload fish image  
- Display uploaded image  
- Predict fish class  
- Show confidence score  
- Display top 5 probabilities  

---

## Run the Application

pip install -r requirements.txt
streamlit run FishImageClassifier.py
Then open the local URL shown in the terminal.
## Project Deliverables

- Trained Models (.h5)  
- Streamlit Web Application  
- Model Evaluation Report  
- Visualization Plots  
- Confusion Matrices  
- Well-structured GitHub Repository  
- Documentation  

---

## Key Highlights

- Handled class imbalance (Focal Loss experimentation)  
- Compared 6 different architectures

---

## Author

Arunkumar S  
Data Science Aspirant  
Passionate about Deep Learning & Model Deployment
