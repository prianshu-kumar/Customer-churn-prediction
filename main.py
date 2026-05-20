from sklearn.model_selection import train_test_split

from src.data_ingestion import load_data
from src.feature_engineering import create_features

from src.model_training import train_all_models
from src.evaluation import evaluate_model

from src.utils import save_object
import pandas as pd
from src.explainability import explain_model
import matplotlib.pyplot as plt
import shap

# Load data
df = load_data(
    "data/processed/cleaned.csv"
)

# Feature engineering
df = create_features(df)

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train all models
trained_models = train_all_models(
    X_train,
    y_train
)

# Store results
results = {}

# Evaluate all
for name, model in trained_models.items():

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    results[name] = metrics

    print(f"\n{name}")
    print(metrics)

# Best model selection
best_model_name = max(
    results,
    key=lambda x: results[x]["roc_auc"]
)

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)

# Save best model
save_object(
    "artifacts/best_model.pkl",
    best_model
)

sample_data = X_test.sample(100, random_state=42)

explainer, shap_values = explain_model(
    best_model,
    sample_data
)

shap.summary_plot(
    shap_values,
    show=True
)