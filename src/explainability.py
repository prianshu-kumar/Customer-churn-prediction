import shap
import pandas as pd


def explain_model(model, X_sample):

    """
    Generate SHAP explanations for model predictions
    """

    # Extract model from pipeline or GridSearchCV
    if hasattr(model, "named_steps"):
        fitted_model = model.named_steps["model"]
        preprocessor = model.named_steps["preprocessor"]
    elif hasattr(model, "best_estimator_"):
        fitted_model = model.best_estimator_.named_steps["model"]
        preprocessor = model.best_estimator_.named_steps["preprocessor"]
    else:
        raise AttributeError("Model does not expose a pipeline via named_steps or best_estimator_.")

    # Transform data using preprocessing
    X_transformed = preprocessor.transform(X_sample)

    # SHAP explainer
    explainer = shap.Explainer(fitted_model, X_transformed)

    shap_values = explainer(X_transformed)

    return explainer, shap_values