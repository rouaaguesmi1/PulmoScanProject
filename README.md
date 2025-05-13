# PneumaTect: Detection and Classification of Pulmonary Nodules

![Logo](https://upload.wikimedia.org/wikipedia/commons/f/ff/Logo_ESPRIT_Ariana.jpg)
> Developed by **NeuralMasters - 4DS3 Team** | ESPRIT University  
> Academic Year: 2024–2025  
> Supervisors: HASNI Fares, ZOUARI Sarra  

---

## 🧠 Project Overview

**PneumaTect** is an AI-powered platform designed to assist in the early detection and classification of pulmonary nodules using advanced deep learning algorithms. The solution aims to improve diagnostic accuracy, reduce radiologist workload, and support clinical decisions in lung cancer management.

Pulmonary nodules may indicate lung cancer—one of the deadliest diseases globally. This platform leverages state-of-the-art medical imaging techniques and AI-driven analysis to classify nodules into normal, benign, or malignant types, and further sub-type lung cancers.

---

## 🎯 Objectives

### 📌 Business Objectives
- Enable **early detection** of lung cancer to improve patient survival rates.
- Reduce diagnostic **time and workload** for radiologists.
- Provide **automated reports** and support for treatment decision-making.
- Promote **scalable and cost-effective** healthcare solutions.
- Align with UN Sustainable Development Goals (SDGs 3, 4, 9, 10, 17).

### 📊 Data Science Objectives
- Detect pulmonary nodules in 3D CT scan data using **3D U-Net with Multi-Scale Attention (CBAM)**.
- Classify nodules using **3D CNN models** into malignant/benign and further into subtypes.
- Predict cancer risk, cancer stage, mortality, and lung function decline.
- Utilize image enhancement pipelines to standardize and preprocess CT data.
- Deploy AI models into an end-to-end system with APIs and UI integration.

---

## 🧬 Datasets

| Dataset                                 | Use Case                                    | Source       |
|----------------------------------------|---------------------------------------------|--------------|
| **LUNA16**                              | Pulmonary Nodule Detection                  | Kaggle       |
| **Data Science Bowl 2017**              | Cancer Prediction from Nodule Detection     | Kaggle       |
| **Chest CT-Scan Dataset**              | Lung Cancer Subtype Classification          | Kaggle       |
| **OSIC Pulmonary Fibrosis Progression**| Lung Function Decline Prediction            | Kaggle       |
| **Lung Cancer Prediction**             | Risk Prediction (Structured Data)           | Kaggle       |
| **Lung Cancer Mortality Dataset**      | Survival Prediction                         | Kaggle       |

---

## 🏗️ Project Architecture

User → UI Interface → PneumaTect API → AI Inference Engine ↳ Detection Model (3D U-Net MSA) ↳ Classification Models (3D CNNs) ↳ Risk/Staging/Mortality Predictors ↳ Report Generator


- **Preprocessing**: Lung segmentation, HU normalization, voxel resampling.
- **Modeling**:
  - *3D U-Net + CBAM* for heatmap-based nodule detection.
  - *3D CNNs* for classification of nodules and cancer stages.
  - *Transfer Learning* (MobileNetV2, ResNet50) for image-based subtype classification.
- **Deployment**: Web and API interface for real-time predictions and visualization.

---

## 🧪 Model Summary

| Task                          | Model Type                  | Dataset        | Highlights                                     |
|-------------------------------|-----------------------------|----------------|-----------------------------------------------|
| Nodule Detection              | 3D U-Net + CBAM             | LUNA16         | Voxel-wise heatmap prediction                 |
| Nodule Classification         | 3D CNN                      | LUNA16         | 32x32x32 patches; BCEWithLogitsLoss           |
| Lung Cancer Detection         | Patient-Level 3D CNN        | DSB 2017       | Whole-volume scan; Malignancy classifier      |
| Subtype Classification        | MobileNetV2, ResNet50       | Chest CT-Scan  | Transfer learning; Early stopping, fine-tune  |
| Lung Function Decline        | Regression CNN+Metadata     | OSIC           | Predict slope in FVC progression              |

---

## 📊 Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- ROC-AUC, Dice Coefficient, Sensitivity
- Validation Loss & False Positive Rates

---

## 🧩 Challenges & Solutions

| Challenge                         | Solution                                                |
|----------------------------------|----------------------------------------------------------|
| Imbalanced Dataset               | SMOTE, Class Weights, Data Augmentation                 |
| High False Positives             | Post-classification filters, thresholding               |
| Storage & Computation Limits     | Federated Learning, A100 GPUs, Cloud Storage            |
| Integration into Clinics         | DICOM Viewer, API Services, EMR Compatibility           |

---

## 🚀 Future Work

- Full training on distributed GPUs (NVIDIA A100).
- Federated learning for hospital-level data privacy.
- Expand into real-time clinical decision support.
- Improve explainability using Grad-CAM, SHAP.
- Integrate multilingual, speech-powered chatbot via DeepSeek LLM.

---

## 👨‍👩‍👧‍👦 Team Members

| Name                    | Role                               |
|-------------------------|------------------------------------|
| Mohamed Amine Ghorbali (AKA Aethelios) | Project Manager / Data Scientist   |
| Rouaa Guesmi           | Data Engineer / Data Scientist     |
| Mohamed Kamel Tounsi   | Application Developer / Data Sci.  |
| Mohamed Mohamed Salem  | Solution Architect / Data Scientist|
| Zeineb Kraiem          | Application Developer / Data Sci.  |

---

## 🏫 Institution

**ESPRIT University**  
Ecole Supérieure Privée d’Ingénierie et de Technologies  
 Engineering Degree

---

## 📁 Project Structure


---

## 📝 License

This project is developed for academic purposes at ESPRIT and may be shared or adapted under the terms of the [MIT License](https://opensource.org/licenses/MIT) for non-commercial use.

---

## 🙌 Acknowledgements

Thanks to our supervisors **HASNI Fares** and **ZOUARI Sarra** for their continuous guidance and feedback. Special thanks to Kaggle and open-source contributors for making medical datasets accessible for research.

---
