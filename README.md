# 🐦 BirdCLEF Baseline Deep Learning Model

## 📌 Overview
This project implements a baseline deep learning pipeline for classifying biological audio recordings into five classes:

- Aves (birds)  
- Mammalia (mammals)  
- Reptilia (reptiles)  
- Amphibia (amphibians)  
- Insecta (insects)  

The model uses **Mel spectrograms + Convolutional Neural Networks (CNNs)**.

---

## 🎯 Task Description
The objective is to classify environmental audio recordings into biological categories using machine learning.

Each audio file is transformed into a Mel spectrogram representation and fed into a CNN model for classification.

---

## 📊 Dataset Overview

- Format: `.ogg`  
- Labels: `class_name`  
- Total samples: ~35,000+  
- Classes:
  - Aves  
  - Mammalia  
  - Reptilia  
  - Amphibia  
  - Insecta  

⚠️ **Severe class imbalance:**
- Aves dominates the dataset  
- Reptilia is extremely underrepresented  

---

## 📥 Dataset Download

The dataset (~9GB) is not included in this repository.

Download here:  
https://drive.google.com/drive/folders/1tJF6oQBQtGvexWWjfHLdcqlNrBon3KBM?usp=sharing

After downloading:
- Place files inside the `train_audio/` directory  
- Ensure `train.csv` is in the root project folder  

---

## 🔍 Exploratory Data Analysis (EDA)

We analyzed:
- Class distribution  
- Audio duration  
- Spectrogram patterns  

### Class-wise Summary

| Class     | Count | Duration Mean | Min Duration | Max Duration |
|----------|------:|--------------:|-------------:|-------------:|
| Insecta  | 199   | 4.87          | 0.096        | 5.0          |
| Reptilia | 1     | 5.00          | 5.00         | 5.0          |
| Amphibia | 451   | 4.76          | 0.036        | 5.0          |
| Mammalia | 99    | 4.44          | 0.087        | 5.0          |
| Aves     | 34799 | 4.92          | 2.74         | 5.0          |

### Key Findings:
- Dataset is highly imbalanced  
- Most clips are ~5 seconds long  
- Aves dominates the dataset  
- Minority classes have significantly fewer samples  

EDA outputs are saved in:
outputs/eda/


---

## ⚙️ Pipeline
CSV → Filter → Sample → Encode → Audio → Mel Spectrogram → Cache → Split → Loader → CNN → Train → Evaluate


---

## 🤖 Model Architecture

Baseline CNN:

- Conv2D → BatchNorm → ReLU → MaxPool  
- Conv2D → BatchNorm → ReLU → MaxPool  
- AdaptiveAvgPool  
- Fully Connected Layer  

**Input:**
- Mel spectrogram (1 × 128 × T)

---

## ⚖️ Handling Class Imbalance

To address class imbalance:
- Class weights are applied in the loss function  
- Helps the model learn minority classes more effectively  

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Macro & Weighted averages  

---

## 📊 Baseline Performance

### Overall Accuracy
Accuracy: 0.612

### Classification Report

| Class     | Precision | Recall | F1-score | Support |
|----------|----------:|------:|---------:|--------:|
| Amphibia | 0.71 | 0.52 | 0.60 | 90 |
| Aves     | 0.61 | 0.75 | 0.68 | 100 |
| Insecta  | 0.57 | 0.65 | 0.60 | 40 |
| Mammalia | 0.31 | 0.25 | 0.28 | 20 |

### Summary Metrics

| Metric        | Precision | Recall | F1-score | Support |
|--------------|----------:|------:|---------:|--------:|
| Macro Avg    | 0.55 | 0.54 | 0.54 | 250 |
| Weighted Avg | 0.62 | 0.61 | 0.61 | 250 |

---

## 📊 Observations

- Model performs best on **Aves** due to high representation  
- Performance on **Mammalia** is lower due to limited samples  
- Overall performance is constrained by class imbalance  

---

## 🚀 How to Run

### 1. Install dependencies

pip install -r requirements.txt

### 2. Run training

python code/BASELINE.py


---

## 🔥 Future Improvements

- Data augmentation (noise injection, time masking, pitch shift)  
- Use of pretrained models (ResNet, EfficientNet)  
- Focal loss for imbalance handling  
- Better sampling strategies  
- Longer temporal context modeling  
