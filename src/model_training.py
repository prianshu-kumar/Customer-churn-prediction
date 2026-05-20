from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.preprocessing import get_preprocessor
from sklearn.preprocessing import LabelEncoder


def get_models():

    models = {

        "logistic_regression": LogisticRegression(
            max_iter=1000
        ),

        "random_forest": RandomForestClassifier(),

        "xgboost": XGBClassifier(
            eval_metric="logloss"
        ),

        "lightgbm": LGBMClassifier(),

        "catboost": CatBoostClassifier(
            verbose=0
        )
    }

    return models


def get_params():

    params = {

        "logistic_regression": {
            "model__C": [0.1, 1, 10]
        },

        "random_forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10]
        },

        "xgboost": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1]
        },

        "lightgbm": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1]
        },

        "catboost": {
            "model__depth": [4, 6],
            "model__learning_rate": [0.01, 0.1]
        }
    }

    return params


def train_all_models(X_train, y_train):

    preprocessor = get_preprocessor(X_train)

    models = get_models()
    params = get_params()

    trained_models = {}

    for name in models:

        print(f"\nTraining {name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", models[name])
        ])

        grid_search = GridSearchCV(
            pipeline,
            params[name],
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )

        # Encode target labels to integers (required by some libraries
        # such as XGBoost). We keep the encoder so we can translate
        # predictions back to the original labels later.
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        grid_search.fit(X_train, y_train_enc)
        grid_search.label_encoder = le

        trained_models[name] = grid_search

        print("Best Params:", grid_search.best_params_)
        print("Best Score:", grid_search.best_score_)

    return trained_models