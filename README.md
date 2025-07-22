# Automatic Detection of Thoracic Pathologies in Chest X-rays with YOLOv8

**Author:** Tomás Vicente  
**Student Number:** 125604  
**Course:** Master’s in Data Science  
**Course Unit:** Deep Learning for Computer Vision  
**Professor:** Professor Tomás Brandão  
**Date:** July 2025

---

## 📌 Objective

To develop an intelligent system based on the YOLOv8-s architecture for the automatic detection of 10 thoracic pathologies (e.g., pneumonia, nodules, cardiomegaly) in chest X-rays.

The system was trained with over 3,000 images from the ChestX-Det10 dataset and evaluated with more than 500 images.

![image](https://github.com/user-attachments/assets/06d63c38-6ae3-4c3c-9fa3-b9c2ac26bc8e)

---

## 📂 Notebook Structure

| Notebook | Description |
|----------|-------------|
| `1_analise_dataset.ipynb` | Exploratory data analysis and class statistics. |
| `2_conversao_yolo.ipynb` | Conversion of annotations to YOLO format. |
| `3_treino.ipynb` | Training the YOLOv8-s model with fine-tuning and optimization strategies. |
| `4_eval.ipynb` | Final model evaluation with standard metrics (mAP, precision, recall). |
| `5_detecao.ipynb` | Application of the trained model to new X-rays for automatic detection. |

---

## 📊 Results

- **Final model:** `y8s_finetune15`
- **Performance on test set (542 images):**
  - `mAP_50:95 = 0.224`
  - `mAP_50 = 0.442`
  - `Recall = 0.446`
  - `Precision = 0.562`
- **Inference time per image:** ~38ms
- **Model size:** ~25MB

The model was selected for maximizing recall—a critical criterion in clinical contexts where false negatives are undesirable.

---

## 🧪 Technologies and Libraries

- **Python 3.10**
- **YOLOv8 - Ultralytics (PyTorch 2.2)**
- **Google Colab**
- Supporting libraries: `numpy`, `pandas`, `opencv`, `matplotlib`, `seaborn`

---

## 📁 Dataset

The dataset used in this project is **ChestX-Det10**, publicly available on Kaggle:

[ChestX-Det10 Dataset – Kaggle](https://www.kaggle.com/datasets/mathurinache/chestxdetdataset)

---

## 📦 Trained Model

The final trained model `y8s_finetune15` can be downloaded here:

https://drive.google.com/drive/folders/10keI22m3eVd9my57UjH3VFRidfMl_Lwz?usp=sharing

---

## 📑 Report

The complete report is available at:  
`Relatorio_APVC_TomasVicente_125604_2025.pdf`

---

## 📬 Contact

For questions or suggestions:  
📧 tomas.vicente.tech@outlook.com
