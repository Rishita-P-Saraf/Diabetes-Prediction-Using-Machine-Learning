# 🩺 Diabetes Prediction Using Machine Learning

This project focuses on building classification models to predict whether a patient is likely to be diagnosed with diabetes based on health-related attributes. Three supervised learning algorithms are implemented and evaluated: **Logistic Regression**, **Naive Bayes**, and **K-Nearest Neighbors (KNN)**.

---

## 📁 Project Structure
```
diabetes-prediction/
│
├── Diabetes Prediction Project- Logistic Regression.ipynb
├── Diabetes Prediction - Naive Bayes.ipynb
├── Diabetes Prediction Project- KNN Classifier.ipynb
├── diabetes.xlsx # Dataset
├── classification_model.pkl # Saved model file
├── Boxplot.jpg # Visualization (e.g., outlier detection)
├── correlation-coefficient.jpg # Correlation heatmap/image
├── README.md # Project documentation
└── .ipynb_checkpoints/ # Jupyter notebook backups
```

---

## 🧪 Dataset Overview

The dataset is stored in `diabetes.xlsx` and includes features related to a patient’s medical profile.

### Features:
- **Pregnancies**
- **Glucose**
- **BloodPressure**
- **SkinThickness**
- **Insulin**
- **BMI**
- **DiabetesPedigreeFunction**
- **Age**
- **Outcome** (target variable: 1 for diabetic, 0 for non-diabetic)

---

## 🎯 Project Goals

- Perform **Exploratory Data Analysis (EDA)** and visualize feature correlations.
- Preprocess and clean the data (handle missing or zero values).
- Train and evaluate multiple **classification models**:
  - Logistic Regression
  - Naive Bayes
  - K-Nearest Neighbors
- Save the best-performing model as a `.pkl` file.

---

## 📘 Model Notebooks

| Notebook | Description |
|----------|-------------|
| **Logistic Regression** | Builds a baseline logistic regression classifier and evaluates it using metrics like accuracy, precision, recall, and ROC AUC. |
| **Naive Bayes** | Implements Gaussian Naive Bayes for classification and compares performance. |
| **KNN Classifier** | Applies K-Nearest Neighbors and tunes `k` to optimize model accuracy. |

---

## 📊 Evaluation Metrics

Each model is evaluated using:
- **Confusion Matrix**
- **Accuracy**
- **Precision, Recall, F1-score**
- **ROC Curve and AUC Score**

Visual tools such as **boxplots** and **correlation heatmaps** were used for EDA (see `Boxplot.jpg`, `correlation-coefficient.jpg`).

---

## 💾 Model Saving

- The best model is serialized and saved as `classification_model.pkl` using `pickle`.

---

## 🧰 Libraries Used

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `pickle` (for model serialization)

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Open any of the Jupyter notebooks:
   ```bash
   jupyter notebook "Diabetes Prediction Project- Logistic Regression.ipynb"
   ```
---
## 🔮 Future Improvements
- Use ensemble methods like Random Forest or XGBoost
- Build a Streamlit web app for interactive predictions
- Perform hyperparameter tuning using GridSearchCV

---
## 📬 Contact
Created by Rishita Priyadarshini Saraf

📧 rishitasarafp@gmail.com

🔗 (https://github.com/Rishita-P-Saraf)
