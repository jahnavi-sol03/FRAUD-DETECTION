# 💳 Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using **Logistic Regression, Random Forest, and XGBoost** on a highly imbalanced dataset.

```text
Transaction → Preprocessing → ML Model → ⚠️ Fraud Probability
```

## 📌 Problem Statement

Credit card fraud detection is a challenging classification problem because fraudulent transactions are extremely rare.

In this dataset, only **0.17% of transactions are fraudulent**. A model that predicts every transaction as legitimate could achieve approximately **99.83% accuracy while detecting zero fraud cases**.

Therefore, this project focuses on **Precision, Recall, F1-score, and ROC-AUC** rather than accuracy alone.

---

## 📊 Dataset

**Credit Card Fraud Detection — Kaggle / ULB**

The dataset contains transactions made by European cardholders in September 2013.

| Property | Value |
|---|---:|
| Total transactions | 284,807 |
| Transactions after removing duplicates | 283,726 |
| Fraud cases after removing duplicates | 473 |
| Fraud percentage | 0.167% |
| Features | `V1`–`V28`, `Time`, `Amount` |
| Target | `Class` |
| Normal class | `0` |
| Fraud class | `1` |

The `V1`–`V28` features are PCA-transformed and anonymised features.

> **Note:** The dataset is not included in this repository because of its size. Download `creditcard.csv` from Kaggle and place it inside the `data/` directory.

---

## 🔧 Project Approach

### 1. Exploratory Data Analysis

The dataset was explored to understand:

- Class distribution
- Transaction amount distribution
- Outliers using boxplots
- Correlation between numerical features
- Distribution of fraudulent and legitimate transactions

### 2. Data Preprocessing

The following preprocessing steps were performed:

- Checked for missing values
- Checked for duplicate records
- Removed duplicate transactions
- Scaled the `Amount` feature using `StandardScaler`

### 3. Train-Test Split

The data was divided into:

- **80% training data**
- **20% testing data**

A **stratified split** was used to preserve the highly imbalanced fraud-to-legitimate transaction ratio in both sets.

### 4. Handling Class Imbalance

Because fraudulent transactions represent only a very small percentage of the dataset, **SMOTE (Synthetic Minority Over-sampling Technique)** was applied to the training data.

SMOTE was applied **only after the train-test split** to prevent data leakage.

The minority class was oversampled to approximately **20% of the majority class** in the training data.

For Random Forest, `class_weight='balanced'` was also used.

### 5. Machine Learning Models

Three classification models were trained and compared:

1. **Logistic Regression** — baseline model
2. **Random Forest** — ensemble tree-based model
3. **XGBoost** — gradient boosting model

### 6. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Feature Importance

---

## 🏆 Results

The test set contained **56,746 transactions**, including **95 fraudulent transactions**.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9962 | 0.274 | 0.768 | 0.404 | 0.922 |
| Random Forest | **0.9994** | **0.886** | 0.737 | **0.805** | 0.945 |
| XGBoost | 0.9992 | 0.765 | **0.789** | 0.777 | **0.968** |

### 📌 Key Findings

- **Accuracy alone is misleading** for highly imbalanced fraud detection problems.
- Logistic Regression achieved **76.8% recall**, but its precision was only **27.4%**, resulting in more false alarms.
- Random Forest achieved the **highest precision (88.6%)** and **highest F1-score (80.5%)**, making it effective when reducing false positives is important.
- XGBoost achieved the **highest recall (78.9%)** and **highest ROC-AUC (96.8%)**, making it useful when detecting as many fraudulent transactions as possible is the priority.
- The choice of model depends on the business cost of **false positives versus missed fraud**.

### 🌟 Most Important Features

The top features identified by the Random Forest model were:

1. `V14`
2. `V17`
3. `V3`
4. `V10`
5. `V4`

---

## 📁 Project Structure

```text
fraud-detection/
│
├── data/
│   └── .gitkeep
│
├── models/
│   ├── fraud_detection_model.pkl
│   └── scaler.pkl
│
├── FRAUD_DETECTION.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

> The original `creditcard.csv` dataset is intentionally excluded from the repository.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `creditcard.csv` from Kaggle and place it inside:

```text
data/creditcard.csv
```

### 5. Run the notebook

```bash
jupyter notebook FRAUD_DETECTION.ipynb
```

> **Note:** Make sure the notebook uses the relative path `data/creditcard.csv` instead of a Google Colab-specific path.

---

## 🔮 Future Improvements

Possible improvements include:

- Threshold tuning using the precision-recall curve
- Cost-sensitive threshold selection based on business requirements
- Cross-validation with SMOTE inside an `imblearn` Pipeline
- Hyperparameter tuning using `RandomizedSearchCV` or Optuna
- SHAP-based model explanations
- Streamlit web application for real-time predictions
- Time-based train-test splitting
- Model monitoring and fraud-pattern drift detection

---

## 🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **imbalanced-learn**
- **Matplotlib**
- **Seaborn**
- **Jupyter Notebook**

---

## 📌 Prediction with the Saved Model

If the trained model and scaler are saved in the `models/` directory, predictions can be made using:

```python
import joblib

model = joblib.load("models/fraud_detection_model.pkl")
scaler = joblib.load("models/scaler.pkl")

row[["Amount"]] = scaler.transform(row[["Amount"]])

fraud_probability = model.predict_proba(row)[0][1]

print("Fraud probability:", fraud_probability)
```

The input `row` should contain the same features used during model training:

```text
Time, V1, V2, ..., V28, Amount
```

---

## ⚠️ Important Notes

- The dataset is highly imbalanced, so accuracy should not be used as the only evaluation metric.
- SMOTE was applied only to the training data.
- The test set was kept untouched for final model evaluation.
- The dataset is not included in this repository because of its size.
- The saved model should only be used with input data having the same feature structure and preprocessing used during training.

---

## 👤 Author

**Your Name**

- 🔗 LinkedIn: `https://linkedin.com/jahnavi-solanki`
- 💻 GitHub: `https://github.com/jahnavi-sol03`
