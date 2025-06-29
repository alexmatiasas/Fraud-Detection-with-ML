import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# fmt: off
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

# fmt: on

# ─────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────


def plot_confusion_matrix_with_metrics(
    y_true,
    y_pred,
    model_name="Model",
    labels=None,
    cmap="Blues",
    save=False,
):

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels
    )
    labels = ["Not Fraud", "Fraud"] if labels is None else labels

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    plt.title(f"Confusion Matrix – {model_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    # Metrics text block
    metrics_text = (
        f"Accuracy: {acc:.3f}\n"
        f"Precision: {prec:.3f}\n"
        f"Recall: {rec:.3f}\n"
        f"F1 Score: {f1:.3f}"
    )
    plt.gcf().text(0.95, 0.5, metrics_text, fontsize=10, va="center")

    if save:
        filename = f"conf_matrix_{model_name.lower().replace(' ', '_')}.png"
        save_plot(filename)
    plt.show()


# ─────────────────────────────────────────────
# ROC Curve
# ─────────────────────────────────────────────
def plot_roc_with_auc(y_true, y_proba, model_name="Model", color="C0", save=False):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color=color, label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {model_name}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save:
        filename = f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
        save_plot(filename)
    plt.show()


# ─────────────────────────────────────────────
# Classification Report
# ─────────────────────────────────────────────
def print_classification_report(y_true, y_pred, model_name="Model"):
    """Prints a clean classification report with a heading."""
    print(f"\n📋 Classification Report – {model_name}")
    print(classification_report(y_true, y_pred))


# ─────────────────────────────────────────────
# Save current plot
# ─────────────────────────────────────────────
def save_plot(filename, folder_relative_to_src="../reports/figures", dpi=300):
    """Saves the current matplotlib figure to the specified relative folder."""
    script_dir = os.path.dirname(__file__)
    target_folder = os.path.normpath(os.path.join(script_dir, folder_relative_to_src))
    os.makedirs(target_folder, exist_ok=True)

    path = os.path.join(target_folder, filename)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"📁 Figure saved to: {path}")


def save_model_metrics_csv(
    model_name,
    y_true,
    y_pred,
    y_proba,
    filepath="../reports/model_metrics.csv",
    overwrite=True,
):
    # Calculate metrics
    acc = round(accuracy_score(y_true, y_pred), 4)
    prec = round(precision_score(y_true, y_pred), 4)
    rec = round(recall_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    auc = round(roc_auc_score(y_true, y_proba), 4)

    row = {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": auc,
    }

    # Create directory if it doesn't exist
    if not os.path.exists(filepath):
        df = pd.DataFrame([row])
    else:
        df = pd.read_csv(filepath)

        # Overwrite existing model metrics if specified
        if overwrite and model_name in df["Model"].values:
            df = df[df["Model"] != model_name]

        # If not, check if the row already exists
        if row in df.to_dict(orient="records"):
            print(f"⚠️ Metrics for '{model_name}' already exist. Skipping.")
            return

        # Add new row
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(filepath, index=False)
    print(f"✅ Metrics saved to: {filepath}")
