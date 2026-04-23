# 🧬 Embryo Development Stage Classification
### CNN (EfficientNet-B0) + LSTM

---

## 📌 Overview

This project focuses on classifying embryo development stages using a deep learning pipeline that combines spatial feature extraction and temporal modeling. The model processes sequences of embryo images and predicts one of **16 developmental phases**, enabling automated and accurate stage recognition.

---

## 🎯 Objectives

- Build a sequence-based classification model
- Capture temporal progression of embryo development
- Handle class imbalance using advanced loss functions
- Evaluate using robust metrics like Accuracy and F1-score

---

## 🧠 Model Architecture

The model follows a **CNN + LSTM** pipeline:

1. **EfficientNet-B0 (CNN)** — Extracts spatial features from each frame
2. **LSTM (Temporal Model)** — Learns sequential dependencies across frames
3. **Fully Connected Layer** — Outputs classification probabilities for 16 phases

---

## 🔄 Pipeline

1. Load frame-level annotations
2. Perform embryo-level data split (Train / Val / Test)
3. Create sliding window sequences (length = 5)
4. Extract features using EfficientNet
5. Feed sequences into LSTM
6. Train using Focal Cross-Entropy Loss
7. Evaluate model performance

---

## 📂 Dataset

- **Source:** [Kaggle Embryo Dataset](https://www.kaggle.com/)
- **Contains:** Embryo image sequences + Phase annotations
- **Total Classes:** 16 developmental stages

---

## ⚙️ Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 96 × 96 |
| Sequence Length | 5 frames |
| Batch Size | 16 |
| Epochs | 5 |
| Learning Rate | 2e-4 |
| Optimizer | AdamW |

---

## 🧪 Training Details

- **Loss Function:** Focal Cross-Entropy
- **Class Imbalance Handling:** Weighted loss
- **Scheduler:** Cosine Annealing
- **Regularization:** Dropout + Data Augmentation

---

## 📊 Evaluation Metrics

- ✅ Accuracy
- ✅ Weighted F1-score
- ✅ Per-class F1-score
- ✅ Confusion Matrix

---

## 📈 Results

- Model successfully learns temporal patterns in embryo development
- Provides reliable classification across multiple stages
- Performance improves with sequence-based learning

---

## 📉 Visualizations

- 📊 Per-class F1 Score Bar Graph
- 📉 Confusion Matrix
- 📈 Training Curves (optional)

---

## 🚀 How to Run

**1. Install dependencies:**
```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn kagglehub
```

**2.** Run the notebook/script step by step

**3. Outputs:**
- Model checkpoint (`.pth`)
- Evaluation metrics
- Visualization plots

---

## 📦 Output Files

| File | Description |
|------|-------------|
| `embryo_cnn_lstm.pth` | Trained model weights |
| `per_class_f1.png` | Per-class performance visualization |
| `confusion_matrix.png` | Prediction analysis heatmap |

---

##  Challenges Faced

- Class imbalance (rare phases like `tHB`)
- Sequential data handling
- High computational requirements

---

## 🔮 Future Improvements

- Use Transformer-based models (ViT + Temporal Attention)
- Increase sequence length
- Apply self-supervised pretraining
- Deploy as a clinical decision support tool

---

## 👨‍💻 Author

**Charukesh Pyla**

---
