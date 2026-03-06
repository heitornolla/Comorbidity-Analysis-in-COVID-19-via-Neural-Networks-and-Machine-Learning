import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import roc_auc_score, roc_curve

from utils.eval_utils import evaluate_model


def compute_auc(model, X_test, y_test, model_name):
    os.makedirs("auc_outputs", exist_ok=True)

    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_scores = model.decision_function(X_test)
    else:
        raise ValueError("Model does not support probability scoring")

    auc = roc_auc_score(y_test, y_scores)
    fpr, tpr, _ = roc_curve(y_test, y_scores)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()

    path = f"auc_outputs/roc_{model_name}.png"

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"AUC ({model_name}): {auc:.4f}")
    print(f"ROC curve saved: {path}")

    return auc


def generate_shap_explanation(
    model, X_train: pd.DataFrame, X_test: pd.DataFrame, model_name: str
):
    os.makedirs("shap_outputs", exist_ok=True)
    try:
        X_test_sample = X_test.sample(min(300, len(X_test)), random_state=42)

        if model_name in ["Random_Forest", "Gradient_Boosting"]:
            explainer = shap.TreeExplainer(model)

        elif model_name in ["Logistic_Regression", "SVM_Linear"]:
            explainer = shap.LinearExplainer(model, X_train)

        else:
            background = shap.sample(X_train, 100)
            explainer = shap.KernelExplainer(model.predict, background)

        shap_values = explainer.shap_values(X_test_sample)

        plt.figure()
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test_sample, show=False)
            shap_matrix = shap_values[1]
        else:
            shap.summary_plot(shap_values, X_test_sample, show=False)
            shap_matrix = shap_values

        plt.title(f"SHAP Summary - {model_name}")
        plot_path = f"shap_outputs/shap_summary_{model_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Feature importance table
        importance = np.abs(shap_matrix).mean(axis=0)

        importance_df = pd.DataFrame(
            {"feature": X_test_sample.columns, "mean_abs_shap": importance}
        )

        importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)

        top10 = importance_df.head(10)
        table_path = f"shap_outputs/shap_importance_{model_name}.csv"
        top10.to_csv(table_path, index=False)
    except Exception as e:
        print(f"SHAP failed for {model_name}: {e}")


def run_ml_pipeline():
    from utils.prepare_dataframes import main as get_data

    train_df, val_df, test_df = get_data()

    X_train = train_df.drop("died", axis=1)
    y_train = train_df["died"]
    X_val = val_df.drop("died", axis=1)
    y_val = val_df["died"]
    X_test = test_df.drop("died", axis=1)
    y_test = test_df["died"]

    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])

    # -1 train, 0 val
    split_indices = ([-1] * len(X_train)) + ([0] * len(X_val))
    pds = PredefinedSplit(test_fold=split_indices)

    models_config = {
        "SVM_Linear": {
            "model": LinearSVC(random_state=42, max_iter=5000),
            "params": {"C": [0.1, 1, 10, 100]},
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
        },
        "Logistic_Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {"C": [0.1, 1, 10]},
        },
        "Gradient_Boosting": {
            "model": HistGradientBoostingClassifier(random_state=42),
            "params": {
                "max_iter": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5],
            },
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {"n_neighbors": [5, 11, 21]},
        },
        "Naive_Bayes": {"model": GaussianNB(), "params": {}},
    }

    for name, config in models_config.items():
        print(f"\n>>> Grid Search: {name}")

        grid = GridSearchCV(
            config["model"], config["params"], cv=pds, scoring="recall", verbose=1
        )
        grid.fit(X_train_val, y_train_val)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        evaluate_model(y_test, y_pred, model_name=f"{name}_Optimized")
        generate_shap_explanation(best_model, X_train, X_test, name)
        auc = compute_auc(best_model, X_test, y_test, name)


if __name__ == "__main__":
    run_ml_pipeline()
