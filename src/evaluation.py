from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    probs = model.predict_proba(X_test)[:, 1]

    le = getattr(model, "label_encoder", None)
    if le is not None:
        y_test_enc = le.transform(y_test)
        preds_enc = preds if preds.dtype.kind in "iu" else le.transform(preds)
    else:
        y_test_enc = y_test
        preds_enc = preds

    metrics = {
        "accuracy": accuracy_score(y_test_enc, preds_enc),
        "precision": precision_score(y_test_enc, preds_enc, pos_label=1 if le is not None else 'Yes'),
        "recall": recall_score(y_test_enc, preds_enc, pos_label=1 if le is not None else 'Yes'),
        "f1": f1_score(y_test_enc, preds_enc, pos_label=1 if le is not None else 'Yes'),
    }

    # roc_auc_score requires binary labels (0/1). If the model has
    # a fitted LabelEncoder, transform `y_test` to encoded form
    # before computing ROC AUC.
    if le is not None:
        y_true_for_roc = y_test_enc
    else:
        y_true_for_roc = y_test

    metrics["roc_auc"] = roc_auc_score(y_true_for_roc, probs)

    return metrics