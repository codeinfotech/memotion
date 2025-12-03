# Hinglish Meme Emotion & Offensiveness Detection  
A Progressive Multimodal Deep Learning System  
Repository: https://github.com/codeinfotech/memotion

---

## 📌 Project Overview  

This repository contains a complete **multimodal deep learning pipeline** for classifying **Hinglish memes** across:

- **Sentiment** → Positive / Neutral / Negative  
- **Emotions** → Humor, Sarcasm, Offensive, Motivational  
- **Intensity levels** (slight, mild, very)  

The project integrates **image understanding**, **OCR-based text extraction**, **Hinglish-aware text processing**, and **deep multimodal fusion**.  
Training was executed progressively through **six iterative notebooks**, each improving upon the limitations of the previous model.

---

## 📂 Repository Structure  

| File / Folder | Description |
|---------------|-------------|
| `dlproject1.ipynb` | Model 1 — Baseline multimodal classifier |
| `dlproject2.ipynb` | Model 2 — Regularization + Basic augmentation |
| `dlproject3.ipynb` | Model 3 — OCR + Hinglish tokenization + heavy augmentation |
| `dlproject4.ipynb` | Model 4 — Deep multimodal architecture + LR scheduler |
| `dlproject5.ipynb` | Model 5 — Hyperparameter tuning |
| `finalprobably.ipynb` | Model 6 — Final optimized multimodal classifier |
| `Progressive-Deep-Learning-for-Meme-Emotion-Analysis.pdf` | Presentation used for project evaluation |
| `final_report.pdf` | Full academic report + extracted training graphs |
| `paper4.pdf / guo2023memotion.pdf` | Reference research papers |
| `README.md` | Documentation (this file) |

---

## 🚀 Motivation  

Classifying Indian memes is harder than traditional sentiment analysis because:

- They contain **Hinglish code-mixed text**  
- Text is usually **embedded inside images** (requires OCR)  
- Emotions like *sarcasm* and *offensiveness* require **cultural context**  
- Memes depend on **text–image contradiction** (image positive + text negative = sarcasm)

This project builds a **culturally aware multimodal deep learning system** that can handle these conditions.

---

## 🧠 System Pipeline (6 Progressive Models)

We follow a unique **iterative training approach**:

### **🔹 Model 1 – Baseline Multimodal Classifier** (`dlproject1.ipynb`)
- Shallow CNN + basic text embedding  
- Simple concatenation fusion  
- Purpose: pipeline sanity check  
- Performance: very low accuracy, heavy underfitting  

---

### **🔹 Model 2 – Regularization Enhancements** (`dlproject2.ipynb`)
- Added dropout & batch normalization  
- Minor augmentations  
- Reduces overfitting but still weak  

---

### **🔹 Model 3 – OCR + Hinglish Tokenization + Strong Augmentation** (`dlproject3.ipynb`)
- OCR-extracted text prioritized  
- Hinglish tokenization for code-mixed slang  
- Strong visual augmentations  
- Significant improvement in 1-off sentiment accuracy (≈0.83)  

---

### **🔹 Model 4 – Advanced Multimodal Fusion Architecture** (`dlproject4.ipynb`)
- Deep CNN backbone  
- Improved text encoder  
- Learning rate scheduler added  
- Massive performance jump  
- Offensive precision reaches **0.85**  

---

### **🔹 Model 5 – Hyperparameter Tuning** (`dlproject5.ipynb`)
- LR sweeps  
- Batch size tuning  
- Dropout/weight decay adjustments  
- Prepares architecture for final optimization  

---

### **🔹 Model 6 – Final Model (Polished & Stable)** (`finalprobably.ipynb`)
- Final fusion architecture  
- Optimal hyperparameters  
- Best overall performance  
- Offensive precision: **0.82–0.86**  
- Stable convergence  

---

## 🏗️ Final Architecture Summary

### **Image Encoder**
- Deep CNN  
- Learns facial expressions, meme layout, visual humor/sarcasm cues  

### **Text Encoder**
- Hinglish-specific tokenization  
- Cleans and embeds OCR text  
- Learns cultural slang + code-mixing patterns  

### **Fusion Layer**
Combines image & text representations to capture:

- Sarcasm (positive image + negative text)  
- Offensiveness (offensive slang + neutral image)  
- Humor (visual exaggeration + funny caption)  

### **Classification Heads**
- Multi-label emotion classifier  
- Sentiment classifier  
- Intensity classifier  

---

## 📈 Key Results

| Metric | Value |
|--------|--------|
| Offensive Precision | **0.82–0.86** |
| Sentiment Accuracy | 0.36–0.38 |
| Macro-F1 | ≈ 0.20 |
| 1-off Sentiment Accuracy | ≈ **0.81–0.83** |
| Validation Loss (best) | ≈ 1.11–1.12 |

The final model performs reliably on nuanced Hinglish memes, especially subtle *sarcasm* and *offensive* cases.

---

## 🧪 Installation & Usage

### **1. Clone the Repository**
```bash
git clone https://github.com/codeinfotech/memotion.git
cd memotion
