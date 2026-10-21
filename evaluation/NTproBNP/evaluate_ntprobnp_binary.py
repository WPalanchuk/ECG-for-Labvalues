import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# Replace with the path to your run folders
BINARY_1LEAD_PATH = "PATH_TO_1LEAD_RUN_FOLDER"

BINARY_12LEAD_PATH = "PATH_TO_12LEAD_RUN_FOLDER"

def evaluate_binary(path, output):
    data = np.load(path + "/preds_test.npz")

    preds = data["preds"]
    targs = data["targs"].squeeze()

    auroc = roc_auc_score(targs, preds[:, 1])

    y_pred = np.argmax(preds, axis=1)

    accuracy = accuracy_score(targs, y_pred)
    f1 = f1_score(targs, y_pred)

    tn, fp, fn, tp = confusion_matrix(targs, y_pred).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    ConfusionMatrixDisplay.from_predictions(
        targs,
        y_pred,
        display_labels=["<300", ">=300"],
        cmap="Blues"
    )

    plt.xlabel("Predicted class")
    plt.ylabel("Reference class")
    plt.tight_layout()

    plt.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    print("N:", len(targs))
    print("AUROC:", auroc)
    print("Accuracy:", accuracy)
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("F1-score:", f1)
    print("Confusion matrix:")
    print(confusion_matrix(targs, y_pred))


print("1-lead")
evaluate_binary(
    BINARY_1LEAD_PATH,
    "confusion_matrix_1lead.pdf"
)

print("\n12-lead")
evaluate_binary(
    BINARY_12LEAD_PATH,
    "confusion_matrix_12lead.pdf"
)