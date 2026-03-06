import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_model(y_true, y_pred, model_name="Model", log_file="results_log.csv"):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
    }

    df_metrics = pd.DataFrame([metrics])

    file_exists = os.path.isfile(log_file)
    df_metrics.to_csv(log_file, mode="a", index=False, header=not file_exists)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=["Not Died", "Died"],
        yticklabels=["Not Died", "Died"],
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Real")
    plt.xlabel("Predicted")

    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig(f"plots/cm_{model_name}.png")
    plt.close()

    return metrics


def get_pytorch_preds(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            outputs = model(batch_X)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    return np.array(all_targets), np.array(all_preds)
