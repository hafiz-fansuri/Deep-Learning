# 🐦 BirdCLEF 2026 — Assessment 3

> Bird species recognition from passive acoustic monitoring using ensemble deep learning.

---

## Overview

This repository documents my approach to the **BirdCLEF 2026 Kaggle competition** as part of Assessment 3. The goal is to identify bird and wildlife species from soundscape recordings across a range of habitats.

Two main methods were explored, progressively improving the public leaderboard score from **0.56 → 0.949**.

---

## Method 1 — Synthetic Audio Augmentation for Missing Species

**Public LB Score: `0.56`**

To handle rare or missing species in the training data, synthetic audio samples were generated using synthesized/augmented soundscapes.

**Key idea:** Use synthetic audio to fill coverage gaps for underrepresented species, then train a baseline classifier on the expanded dataset.

**Limitations observed:**
- Synthetic audio introduces domain mismatch — the model trained well on artificial sounds but generalised poorly to real-world passive recordings.
- Low recall on rare species despite augmentation.

---

## Method 2 — Pretrained Model Ensemble

**Public LB Score: `0.90 → 0.949`**

Switched to leveraging publicly available high-scoring pretrained notebooks and ensembling their predictions. Three top-performing public solutions were selected and combined.

### Models Used

| Model ID | LB Score | Version | Notebook Title | Author | Tier |
|----------|----------|---------|----------------|--------|------|
| Model 22 | 0.928 | v.18 | Bird26.REPRODUCE.Perch+ ProtoSSM+ResSSM.INF/TRAIN | yukiZ | 🏆 Grandmaster |
| Model 52 | 0.949 | v.1 | birdclef 2026 exp019 eos4 rank power 06 | Derek | Contributor |
| Model 73 | 0.949 | v.6 | v6\_0949\_replay | Yaroslav Kholmirzayev | Expert |

### Ensemble Strategy

Predictions from **Model 22, Model 52, and Model 73** were combined via soft voting (probability averaging) across all species classes.

- Model 22 uses **Google Perch embeddings** with ProtoSSM and ResSSM architectures — strong generalization across species.
- Model 52 achieves top-tier single-model performance with a rank-power ensemble approach.
- Model 73 is a replay-based refinement of a 0.949-scoring submission, providing complementary prediction diversity.

**Result:** The ensemble improves robustness by reducing individual model variance while retaining peak scores.

---

## Results Summary

| Method | Approach | Public LB Score |
|--------|----------|----------------|
| Method 1 | Synthetic audio for missing species | 0.56 |
| Method 2 (ensemble: Model 22 + 52 + 73) | Soft voting ensemble | **0.90** |

---

## Environment

- **Platform:** Kaggle Notebooks (GPU T4 x2)
- **Python:** 3.12
- **Key libraries:** `torch`, `timm`, `librosa`, `numpy`, `pandas`, `sklearn`
- **Local env:** Windows, conda `MV1`

---

## Competition

- **Competition:** [BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026)
- **Task:** Multi-label bird species classification from 5-second audio chunks
- **Metric:** macro-averaged ROC-AUC

---


Mechatronics Engineering, IIUM  
GitHub: [@hafizfansuri](https://github.com/hafizfansuri)
