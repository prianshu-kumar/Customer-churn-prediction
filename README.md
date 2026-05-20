# 📉 Customer Churn Prediction System (End-to-End ML Project)

An **industry-grade machine learning system** that predicts customer churn using advanced ML pipelines, hyperparameter tuning, and model explainability (SHAP). The project is designed with a **production-style architecture**, including modular code, reusable pipelines, and a deployed Streamlit dashboard.

---

# 🚀 Live Features

- 📊 Customer churn prediction (binary classification)
- ⚙️ End-to-end ML pipeline (preprocessing + modeling)
- 🤖 Multiple ML models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost)
- 🔥 Hyperparameter tuning using GridSearchCV
- 📈 Model evaluation using ROC-AUC, F1, Precision, Recall
- 🧠 Feature engineering for business insights
- 🔍 Model explainability using SHAP
- 🌐 Interactive Streamlit dashboard
- 💾 Model persistence using pickle

---

# 🏗️ Project Architecture

```text
customer-churn-prediction/
│
├── app/                      # Streamlit web app
│   └── app.py
│
├── artifacts/                # Saved models & outputs
│   ├── best_model.pkl
│   └── insights.txt
│
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/           # Cleaned dataset
│
├── notebook/                # EDA & experiments
│   └── eda.ipynb
│
├── src/                     # Production ML code
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── utils.py
│
├── main.py                  # Training pipeline entry point
├── requirements.txt
├── README.md
└── setup.py
```

---

# 📊 Dataset Information

This project uses a telecom customer dataset containing:

### Features:
- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- MonthlyCharges
- TotalCharges
- Contract type
- Payment method

### Target:
- `Churn` (Yes / No)

---

# 🧠 Machine Learning Pipeline

The system follows a **fully automated ML pipeline**:

```text
Raw Data
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
ColumnTransformer (Encoding + Scaling)
   ↓
Model Training (Multiple Algorithms)
   ↓
Hyperparameter Tuning (GridSearchCV)
   ↓
Best Model Selection
   ↓
Evaluation (ROC-AUC, F1, Precision, Recall)
   ↓
Explainability (SHAP)
   ↓
Model Deployment (Streamlit App)
```

---

# ⚙️ Tech Stack

### Programming
- Python

### Data Handling
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

### Visualization
- Matplotlib
- Seaborn
- SHAP

### Deployment
- Streamlit

---

# 🔥 Key Features (What makes this project advanced)

## 1. Modular ML Architecture
Clean separation of:
- preprocessing
- training
- evaluation
- inference

## 2. Multi-Model Training System
Automatically compares:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

## 3. Hyperparameter Optimization
Uses GridSearchCV to select best model configuration.

## 4. Feature Engineering
Creates business-driven features like:
- Average Monthly Spend
- Customer tenure risk groups
- Charge behavior patterns

## 5. Explainable AI (SHAP)
- Global feature importance
- Local prediction explanation
- Business interpretability

## 6. Production-Ready Deployment
Streamlit app with:
- real-time prediction
- probability scores
- business insights dashboard

---

# 📈 Model Evaluation Metrics

The system evaluates models using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC (primary metric)

---

# 📊 Business Insights

Key churn drivers identified:

- High Monthly Charges → higher churn risk
- Month-to-month contracts → unstable customers
- Low tenure → early churn stage
- Payment method behavior impacts retention

---

# 🌐 Streamlit App Features

The deployed app includes:

### 🔮 Prediction Module
- Input customer details
- Get churn prediction
- View probability score

### 📊 Explainability Module
- SHAP global feature importance
- Individual prediction explanation

### 💼 Business Dashboard
- Key churn insights
- Retention strategy recommendations

---

# 🚀 How to Run Locally

## 1. Clone repository
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

## 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
```

## 3. Install dependencies
```bash
pip install -r requirements.txt
```

## 4. Train model
```bash
python main.py
```

## 5. Run Streamlit app
```bash
streamlit run app/app.py
```

---

# 📦 Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
catboost
streamlit
shap
```

---

# 🔥 Future Improvements

- Add MLflow experiment tracking
- Docker containerization
- FastAPI backend for predictions
- CI/CD pipeline using GitHub Actions
- Real-time model monitoring
- Cloud deployment (AWS / Azure)

---

# 👨‍💻 Author

**Prianshu Kumar**

- GitHub: https://github.com/prianshu-kumar  
- Domain: Machine Learning | Data Science | AI Systems

---

# ⭐ Project Highlights

✔ Industry-style ML architecture  
✔ Multi-model experimentation system  
✔ Hyperparameter tuning pipeline  
✔ SHAP explainability integration  
✔ Full Streamlit deployment  
✔ Business-oriented insights  

---

# 🚀 Summary

This project demonstrates a **real-world machine learning system** that goes beyond basic model building and includes:

> Data engineering + ML pipelines + model selection + explainability + deployment

---
