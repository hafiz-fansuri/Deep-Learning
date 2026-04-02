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

Each audio file is transformed into a spectrogram representation and fed into a CNN model.

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
- Aves dominates dataset
- Reptilia extremely rare

---

## 📥 Dataset Download

Dataset (~9GB) is not included.

👉 Download here: **https://drive.google.com/drive/folders/1weWJeCExWQS6FhUqsT8BjlV9nQw5i2HM?usp=sharing**

After downloading:
---

## 🔍 Exploratory Data Analysis (EDA)

We analyzed:
- Class distribution
- Audio duration
- Spectrogram patterns

### Key Findings:
- Dataset heavily imbalanced
- Most clips ~5 seconds
- Classes differ in frequency patterns

📁 Outputs saved in:
outputs/eda/

---

## ⚙️ Pipeline
<p align="center">
  <img src="pipeline.png" width="600">
</p>

---

## 🤖 Model Architecture

Baseline CNN:

- Conv2D → ReLU → MaxPool
- Conv2D → ReLU → MaxPool
- AdaptiveAvgPool
- Fully Connected Layer

Input:
- Mel spectrogram (1 × 64 × T)

---

## ⚖️ Handling Class Imbalance

We use **class weighting**:

- Rare classes get higher loss weight
- Helps model learn underrepresented classes

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score

---

## 📊 Baseline Performance


| Class     | Precision | Recall | F1-score |
|----------|----------|--------|---------|
| Aves     | 0.85     | 0.88   | 0.86    |
| Mammalia | 0.72     | 0.70   | 0.71    |
| Amphibia | 0.68     | 0.66   | 0.67    |
| Insecta  | 0.65     | 0.63   | 0.64    |
| Reptilia | 0.10     | 0.05   | 0.07    |

Overall Accuracy: **0.75**

---

## 🚀 How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run training
python src/train.py

---

## 🔥 Future Improvements

- Data augmentation (noise, pitch shift)
- Better architectures (ResNet, EfficientNet)
- Focal loss for imbalance
- Longer audio context

---


