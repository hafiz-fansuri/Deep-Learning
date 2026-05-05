# Assessment 2

## BirdCLEF 2026 – Google Perch Baseline Pipeline

---

## 1. Project Overview

This project implements a baseline audio classification pipeline for the **BirdCLEF 2026 competition** using **Google Perch v2**, a pretrained bird vocalization model.

The system processes environmental soundscape recordings and predicts the presence of bird species based on short audio segments.

---

## 2. Objective

* Classify bird species from audio recordings
* Use pretrained **Google Perch v2 model** without fine-tuning
* Generate submission file for Kaggle BirdCLEF 2026
* Handle strict runtime constraint (~1 hour)

---

## 3. Dataset

### Input Data:

* `.ogg` soundscape audio files
* BirdCLEF taxonomy mapping (`taxonomy.csv`)
* Sample submission format (`sample_submission.csv`)

### Labels:

* 234 target bird species
* Scientific names mapped to common labels

---

## 4. Model Used

### Google Perch v2 (TensorFlow SavedModel)

* Pretrained bioacoustic classifier
* Outputs probability distribution over large species set
* Not originally aligned with BirdCLEF taxonomy

---

## 5. System Pipeline

### Step 1: Environment Setup

* Install TensorFlow 2.20 (required for model compatibility)
* Load required libraries (librosa, pandas, numpy, tensorflow)

---

### Step 2: Load Model

```python
birdclassifier = tf.saved_model.load('perch_v2_cpu')
```

---

### Step 3: Label Mapping

* Load Perch label list (scientific names)
* Join with BirdCLEF taxonomy
* Map:

  ```
  primary_label → bc_index (Perch output index)
  ```
* Handle missing species by assigning fallback index

---

### Step 4: Audio Loading

* Load `.ogg` files using `librosa`
* Resample to **32,000 Hz**

---

### Step 5: Segmentation

* Audio split into **5-second windows**
* Each segment represents one prediction step

---

### Step 6: Inference

```python
model_outputs = birdclassifier.signatures['serving_default'](
    inputs=audio.reshape((-1, 5 * sr))
)['label']
```

* Apply model on each chunk
* Extract probability scores

---

### Step 7: Post-processing

* Pad outputs for missing class alignment
* Select BirdCLEF-relevant indices only
* Map predictions to `primary_labels`

---

### Step 8: Parallel Execution

* Use `ThreadPoolExecutor(max_workers=4)`
* Process multiple soundscapes simultaneously

---

### Step 9: Submission Generation

* Combine results into dataframe
* Format:

  ```
  row_id + species probability columns
  ```
* Export as `submission.csv`

---

## 6. Key Hyperparameters

| Parameter      | Value         |
| -------------- | ------------- |
| Sample Rate    | 32000 Hz      |
| Segment Length | 5 seconds     |
| Threads        | 4             |
| Runtime Limit  | ~5300 seconds |

---

## 7. Key Challenges

### 1. Runtime Limitation

* Perch inference is computationally heavy
* Full dataset processing is not feasible within Kaggle time limit

### 2. Species Coverage Gap

* Perch model does not include all BirdCLEF species
* Only subset of 234 species can be reliably predicted

### 3. Audio Processing Bottleneck

* Full `.ogg` loading is slow
* Large memory usage when processing long soundscapes

---

## 8. Techniques Tried

### ✔ Baseline Perch inference

* Functional but too slow for full dataset

### ✔ Taxonomy mapping

* Linked scientific names to BirdCLEF labels
* Improved compatibility with submission format

### ✔ Parallel processing

* Improved speed using multithreading
* Limited by CPU constraints

---

## 9. Key Insights

* Model accuracy is secondary to **runtime efficiency** in this competition
* Coverage of soundscapes impacts leaderboard score more than fine-tuning
* Perch model is powerful but not optimized for full BirdCLEF taxonomy

---

## 10. Future Improvements

### Planned Enhancements:

* Chunked streaming audio processing (reduce memory load)
* Silent segment skipping (remove unnecessary inference)
* Reduce thread count for stability (4 → 2)
* Use float16 inference optimization
* Explore hybrid model (Perch embeddings + lightweight classifier)

---

## 11. Conclusion

This project demonstrates a working baseline implementation of Google Perch for BirdCLEF 2026. While the model performs well in inference quality, system-level optimizations are required to meet competition constraints and improve leaderboard coverage.

---

## 12. File Output

* `submission.csv` generated in Kaggle format
* Compatible with BirdCLEF 2026 evaluation system
