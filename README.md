# 🧠 Brain MRI Tumor Segmentation (U-Net)

A deep learning project for **brain tumor segmentation** from MRI images using a custom **U-Net architecture** built with PyTorch.
The project includes **training, evaluation, and a Streamlit web app for inference**.

---

## 🚀 Project Overview

* **Task:** Binary image segmentation (Tumor vs Background)
* **Dataset:** LGG MRI Segmentation Dataset
* **Model:** Custom lightweight U-Net
* **Framework:** PyTorch
* **Deployment:** Streamlit Web App

👉 This project focuses on **data-centric improvements** rather than increasing model complexity.

---

## 🧠 Key Features

* ✅ Custom U-Net (optimized & lightweight)
* ✅ Strong data augmentation (Albumentations)
* ✅ Combined loss (BCE + Dice)
* ✅ Metrics: Dice Score & IoU
* ✅ Early stopping + LR scheduler
* ✅ Streamlit UI for real-time prediction

---

## 📂 Project Structure

```
├── model.py          # U-Net architecture
├── utils.py          # preprocessing & metrics
├── app.py            # Streamlit app
├── best_model.pth    # trained weights
├── notebook.ipynb    # full training pipeline
├── README.md
```

---

## ⚙️ Pipeline

### 1. Data Loading

* MRI images and masks are loaded from dataset
* Matched using filename patterns

### 2. Data Splitting

* Train: 70%
* Validation: 15%
* Test: 15%

---

### 3. Data Augmentation

Applied only on training:

* Flip (Horizontal & Vertical)
* Rotation & Shift
* Elastic & Grid distortions
* Brightness & Contrast
* Gaussian Noise
* Normalization (ImageNet stats)

---

### 4. Model Architecture (U-Net)

#### 🔹 Encoder

* 4 convolution blocks
* Each followed by max pooling

#### 🔹 Bottleneck

* ConvBlock (512 → 512)
* Dropout (0.3)

#### 🔹 Decoder

* Transposed convolution (upsampling)
* Skip connections
* Size alignment fix using interpolation

#### 🔹 Output

* Final Conv layer → 1 channel (binary mask)

---

### ⚠️ Design Decision

Instead of standard 5-block U-Net:

* Used **4 blocks**
* Reduced parameters
* Faster training
* Lower overfitting risk

---

## 📉 Loss Function

Combined loss:

* **BCEWithLogitsLoss** → pixel accuracy
* **Dice Loss** → overlap quality

```
Total Loss = BCE + Dice
```

---

## 📊 Evaluation Metrics

* **Dice Score**
* **IoU (Intersection over Union)**

---

## 🏋️ Training Setup

* Optimizer: `AdamW`
* Learning Rate: `3e-4`
* Scheduler: `ReduceLROnPlateau`
* Gradient Clipping: `1.0`
* Early Stopping: patience = 10

---

## 📈 Results

| Split | Loss  | Dice  | IoU   |
| ----- | ----- | ----- | ----- |
| Train | 0.111 | 0.900 | 0.823 |
| Valid | 0.119 | 0.894 | 0.811 |
| Test  | 0.121 | 0.892 | 0.809 |

### 🔍 Observations

* Stable training
* Minimal overfitting
* Strong generalization

---

## 🖼️ Visualization

The model outputs:

* Original MRI image
* Ground truth mask
* Predicted mask

---

## 🌐 Streamlit App

Run locally:

```bash
streamlit run app.py
```

### Features:

* Upload MRI image
* View predicted tumor mask
* Overlay visualization (green tumor area)

---

## 🔄 Inference Pipeline

1. Resize image → 256×256
2. Normalize (ImageNet mean/std)
3. Model prediction
4. Apply sigmoid
5. Threshold (0.35)
6. Resize back to original

---

## 📌 Important Notes

* ⚠️ Same normalization must be used in training & inference
* ⚠️ Threshold tuning (0.35) improves segmentation
* ⚠️ Interpolation fix prevents size mismatch in decoder

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Albumentations
* OpenCV
* NumPy
* Streamlit

---

## 🎯 Key Insight

> Improving **data quality and augmentation** can be more effective than increasing model complexity — especially in medical imaging.

---

## 📎 Reference

Original notebook & full pipeline:
