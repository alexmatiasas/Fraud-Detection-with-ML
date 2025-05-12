import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report
)

# ─────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix_with_metrics(
    y_true, y_pred, model_name="Model", labels=["Not Fraud", "Fraud"], cmap="Blues", save=False
):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels)

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
    metrics_text = f"Accuracy: {acc:.3f}\nPrecision: {prec:.3f}\nRecall: {rec:.3f}\nF1 Score: {f1:.3f}"
    plt.gcf().text(0.95, 0.5, metrics_text, fontsize=10, va='center')

    if save:
        filename = f"conf_matrix_{model_name.lower().replace(' ', '_')}.png"
        save_plot(filename)
    plt.show()

# ─────────────────────────────────────────────
# ROC Curve
# ─────────────────────────────────────────────
def plot_roc_with_auc(
    y_true, y_proba, model_name="Model", color="C0", save=False
):
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
    model_name, y_true, y_pred, y_proba,
    filepath="../reports/model_metrics.csv", overwrite=True
):
    # Calcular métricas
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
        "ROC AUC": auc
    }

    # Crear archivo si no existe
    if not os.path.exists(filepath):
        df = pd.DataFrame([row])
    else:
        df = pd.read_csv(filepath)

        # ¿Sobrescribir fila existente con mismo nombre?
        if overwrite and model_name in df["Model"].values:
            df = df[df["Model"] != model_name]

        # ¿Ya existe la fila exacta? No hacer nada
        if row in df.to_dict(orient="records"):
            print(f"⚠️ Metrics for '{model_name}' already exist. Skipping.")
            return

        # Agregar nueva fila
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(filepath, index=False)
    print(f"✅ Metrics saved to: {filepath}")