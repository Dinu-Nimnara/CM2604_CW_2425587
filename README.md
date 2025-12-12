# CM2604_CW_2425587 (Coursework - 2425587)
This repository contains the implementation and documentation for the CM2604 Machine Learning coursework. The project involves preparing a dataset, performing exploratory data analysis, developing classification models (Neural Network and Decision Tree), evaluating results, and discussing AI ethics using the Telco Customer Churn dataset.
---

## 📖 Project Overview

This repository contains the full implementation for the CM2604 Machine Learning coursework.  
The project focuses on preparing the Telco Customer Churn dataset, performing exploratory data analysis (EDA), engineering features, training machine learning models, evaluating them, and addressing AI ethics considerations.

The primary objective is to build a model that predicts whether a customer is likely to **churn** based on service usage, demographics, contract types, and billing behaviour.

Two machine learning models were developed and compared:

- **Optimized Decision Tree** (with pruning and hyperparameter tuning)  
- **Deep Neural Network (DNN)** (with regularization and batch normalization)

---

## 📉 Telco Customer Churn Prediction

The model identifies at-risk customers by analyzing:

- Demographic information  
- Service subscription patterns  
- Contract duration and payment methods  
- Monthly billing behaviour  
- Customer tenure

---

## 🛠️ Technologies Used

- Python 3.10+
- TensorFlow / Keras
- Scikit-Learn
- Pandas & NumPy
- Matplotlib & Seaborn
- Google Colab
- Joblib & Keras-Tuner

---

## 📊 Dataset

**Source:** Kaggle – *Telco Customer Churn Dataset*

**Important attributes include:**

- `tenure` — number of months the customer has stayed  
- `MonthlyCharges` — monthly billed amount  
- `Contract` — Month-to-month, One-year, Two-year  
- `PaymentMethod` — customer payment method  
- `Churn` — target (Yes/No)

---

# 🚀 Methodology

## 1. Data Preprocessing
- Cleaned missing values (`TotalCharges`)  
- Corrected datatypes  
- Feature engineering:  
  - Created `tenure_group` (used in EDA)  
  - One-Hot Encoding for multi-class categorical features  
  - Label Encoding for binary features  
- Scaled numerical features using **StandardScaler** (for NN stability)  
- Handled class imbalance using **class_weight='balanced'**

---

## 2. Decision Tree (Optimized & Pruned)

- Performed **GridSearchCV** with 5-fold cross-validation  
- Tuned:  
  - `max_depth`  
  - `criterion` (Gini / Entropy)  
  - `min_samples_split`  
  - `ccp_alpha` (cost-complexity pruning)  
- Result: robust, interpretable, pruned tree

---

## 3. Neural Network (Tuned DNN)

### Architecture

Input Layer
↓
Dense (L2 Regularization) → Batch Norm → ReLU → Dropout
↓
Dense (L2 Regularization) → Batch Norm → ReLU → Dropout
↓
Sigmoid Output Layer



### Training Strategy
- Hyperparameter tuning with **KerasTuner**
- **EarlyStopping** to reduce overfitting  
- Batch normalization for training stability  
- Regularization for generalization

---

# 📈 Model Performance

| Metric | Decision Tree | Neural Network |
|--------|----------------|----------------|
| **Accuracy** | ~74% | ~76% |
| **ROC–AUC** | ~0.83 | ~0.85 |
| **Recall (Churn)** | >80% | >80% |

### Key Insights
- Neural Network outperformed the Decision Tree in AUC and stability  
- Most influential churn indicators:  
  - **Month-to-month contracts**  
  - **High monthly charges**  
- Models exported for deployment:  
  - `churn_dt_model.pkl`  
  - `churn_nn_model.keras`  
  - `scaler.pkl`

---

# 💻 Installation & Usage

## 1. Clone the Repository
bash
git clone https://github.com/your-username/telco-churn-prediction.git
cd telco-churn-prediction

## 2. Install Dependencies

pip install -r requirements.txt

or manually:

pip install pandas numpy scikit-learn tensorflow matplotlib seaborn joblib keras-tuner

## 3. Run the Notebook
jupyter notebook CM2604_CW_2425587.ipynb

# 📂 Project Structure

├── CM2604_CW_2425587.ipynb       # Final notebook (correct file)
├── models/                       # Saved ML models
│   ├── churn_dt_model.pkl
│   ├── churn_nn_model.keras
│   └── scaler.pkl
├── README.md                     # Project documentation

# 📝 Note on File Naming

During initial development in Google Colab, the notebook was accidentally saved as:

CM1604_CW_2425587.ipynb (incorrect)

A corrected version was later created:

CM2604_CW_2425587.ipynb (final & correct)

The incorrect file has been removed to keep the repository clean.
All analysis, training, and evaluation should be reviewed in the updated notebook.


👤 Author Details

Name: N. H. Dinugi Nimnara
Student RGU ID: 2425587
Student ID: 20231363
Module: CM2604 – Machine Learning
