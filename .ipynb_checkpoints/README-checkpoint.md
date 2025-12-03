
# 🧬 Breast Cancer Classification — End-to-End Machine Learning Pipeline

A full end-to-end machine learning project using the **Breast Cancer Wisconsin (Diagnostic) Dataset**, implementing:

- Automated preprocessing  
- Feature engineering  
- Feature selection  
- Model training  
- Hyperparameter tuning  
- Explainability with SHAP  
- Error analysis  
- Production-ready inference pipeline (`predict.py`)  

This project follows a modular, industry-grade structure suitable for real deployment and serves as a strong portfolio example for Data Science / ML Engineering positions.

---

## 📌 1. Project Objective

The goal of this project is to build a robust and explainable machine learning model capable of classifying tumors as **benign** or **malignant** based on computed radiological measurements of cell nuclei.

Special attention is given to:

- Interpretability (SHAP)
- Clinical reliability (False Negatives)
- Production-ready inference
- Proper ML project structure

---

## 📊 2. Dataset

- **Source:** Kaggle — Breast Cancer Wisconsin (Diagnostic)  
  https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

- **Rows:** 569  
- **Features:** 30 numeric predictors  
- **Target:**  
  - `M` → Malignant  
  - `B` → Benign  

No missing values are present. Several features exhibit strong multicollinearity and tumor-related outliers, which were investigated but **not removed** because they represent clinically meaningful cases (mostly malignant).

---

## 🧱 3. Project Architecture

```
project_root/
│
├── data/
│   ├── raw/               # Original dataset
│   └── processed/         # Preprocessed + split + engineered features
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   ├── 04_Feature_Selection.ipynb
│   ├── 05_Hyperparameter_Tuning.ipynb
│   ├── 06_Model_Evaluation.ipynb
│   └── 07_Testing_Inference.ipynb
│
├── src/
│   ├── config.py
│   ├── data/
│   │   └── preprocessing.py
│   ├── models/
│   │   └── final_model.pkl
│   └── inference/
│       └── predict.py     # Inference pipeline (production-ready)
│
└── requirements.txt
```

---

## ⚙️ 4. Technologies Used

- Python 3.10+
- scikit-learn
- pandas / numpy
- seaborn / matplotlib
- plotly
- SHAP
- joblib
- JupyterLab

---

## 🚀 5. Installation

### **1. Clone the repository**
```bash
git clone https://github.com/<your-username>/breast-cancer-classification.git
cd breast-cancer-classification
```

### **2. Create environment (optional but recommended)**  
If you use Anaconda:

```bash
conda create -n cancer python=3.10
conda activate cancer
```

### **3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 🧼 6. Preprocessing Overview

The preprocessing pipeline applies:

- Label encoding (`diagnosis` → 0/1)
- Standard scaling (train-set-only)
- Feature engineering (e.g., `radius_avg`, `area_avg`, etc.)
- Variance-related noise reduction
- Train/test split
- Saving:
  - `X_train_preprocessed.csv`
  - `X_test_preprocessed.csv`
  - `y_train.csv`
  - `y_test.csv`
  - `scaler.pkl`

Outliers were **not removed**, as they represent clinically meaningful malignant cases.

---

## 🧠 7. Feature Selection Summary

Techniques used:

- Mutual Information  
- ANOVA (f-score)  
- Tree-based importance (Random Forest)  
- Embedded methods (Gradient Boosting)  
- SHAP feature contributions  

The final set of selected features was chosen based on **multi-criteria agreement**, model performance, and clinical interpretability.

---

## 🔥 8. Model Training & Hyperparameter Tuning

Several classifiers were tested:

- Logistic Regression  
- SVM  
- Random Forest  
- KNN  
- Gradient Boosting (final model)

The final model was selected using:

- Stratified train/test split  
- GridSearchCV  
- Evaluation via:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - ROC AUC  

### **Final Model:**  
✔ **GradientBoostingClassifier** (best generalization + best SHAP stability)

---

## 📈 9. Model Evaluation (Step 6 Overview)

### **Main metrics (final test set):**
- **Accuracy:** 97.36%  
- **Precision:** 100%  
- **Recall:** 92.85%  
- **F1-score:** 96.29%  
- **ROC-AUC:** 99.23%  


### Visual outputs:
- Confusion Matrix  
- ROC Curve  
- Precision–Recall Curve  
- Classification Report (formatted as DataFrame)

---

## 🧩 10. Explainability (SHAP)

The model was fully explained using SHAP:

- **Beeswarm Plot** — global feature importance  
- **Bar Plot** — average absolute SHAP contribution  
- **Dependence Plots** — feature interaction patterns  
- **Waterfall Plot** — local explanation of individual predictions  

This ensures interpretability and clinical trust.

---

## 🔍 11. Error Analysis

Both **False Positives** and **False Negatives** were examined in detail.

- Most false negatives occur in borderline malignant cases → clinically critical  
- False positives tend to appear in high-variance benign tumors  
- This analysis reinforces model trustworthiness and highlights risk boundaries

---

# 🧪 12. Inference — How to Use the Prediction Pipeline

The inference pipeline (`predict.py`) loads:

- the trained model  
- the saved scaler  
- the selected features  

### **Single sample prediction**

```python
from src.inference.predict import PredictionPipeline

pipeline = PredictionPipeline()

sample = {
    "radius_mean": 14.1,
    "texture_mean": 20.3,
    ...
}

pipeline.predict_single(sample)
```

Output (DataFrame):

| feature1 | feature2 | … | prediction | prediction_label | probability_malignant |
|----------|----------|----|------------|-------------------|------------------------|
| …        | …        | …  | 1          | Malignant         | 0.982                 |

---

### **Batch prediction**

```python
df = pd.read_csv("new_samples.csv")
pipeline.predict_batch(df)
```

---

## 📦 13. Running the Full Pipeline

All steps are contained in the `notebooks/` directory, following a clean modular structure:

- **01 → EDA**  
- **02 → Preprocessing**  
- **03 → Modeling**  
- **04 → Feature Selection**  
- **05 → Hyperparameter Tuning**  
- **06 → Model Evaluation**  
- **07 → Testing Inference**  

---

## 🛠️ 14. Future Work

- API deployment with FastAPI  
- Dockerization  
- Cross-validation with nested CV  
- SMOTE / class balancing experiments  
- Model monitoring and drift detection  
- MLflow experiment tracking  

---

## 🧑‍🔬 Author

**Mateus Vieira Vasconcelos**  
Data Science & Machine Learning Enthusiast  

---
