import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    ConfusionMatrixDisplay
)

# Replace with the path to your run folders
CE_PATH = "PATH_TO_CE_RUN_FOLDER"

WEIGHTED_CE_PATH = "PATH_TO_WEIGHTED_CE_RUN_FOLDER"

WEIGHTED_CEF_PATH = "PATH_TO_WEIGHTED_CEF_RUN_FOLDER"

def evaluate_fourclass(path):
    data = np.load(path + "/preds_test.npz")

    preds = data["preds"]
    targs = data["targs"].squeeze()

    y_pred = np.argmax(preds, axis=1)

    accuracy = accuracy_score(targs, y_pred)

    sensitivity = recall_score(
        targs,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        targs,
        y_pred,
        average="macro"
    )

    cm = confusion_matrix(targs, y_pred)

    print("N:", len(targs))
    print("Accuracy:", accuracy)
    print("Sensitivity macro:", sensitivity)
    print("F1-score macro:", f1)
    print("Confusion matrix:")
    print(cm)


def make_confusion_matrix(path, output_file):
    data = np.load(path + "/preds_test.npz")

    preds = data["preds"]
    targs = data["targs"].squeeze()

    y_pred = np.argmax(preds, axis=1)

    ConfusionMatrixDisplay.from_predictions(
        targs,
        y_pred,
        display_labels=["<300", "300-1000", "1000-2000", ">2000"],
        cmap="Blues"
    )

    plt.xlabel("Predicted class")
    plt.ylabel("Reference class")
    plt.tight_layout()

    plt.savefig(
        output_file,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()


print("Cross-entropy")
evaluate_fourclass(CE_PATH)

print("\nWeighted cross-entropy")
evaluate_fourclass(WEIGHTED_CE_PATH)

print("\nWeighted focal cross-entropy")
evaluate_fourclass(WEIGHTED_CEF_PATH)


make_confusion_matrix(
    CE_PATH,
    "confusion_matrix_nyha_ce.pdf"
)

make_confusion_matrix(
    WEIGHTED_CE_PATH,
    "confusion_matrix_nyha_weighted_ce.pdf"
)

make_confusion_matrix(
    WEIGHTED_CEF_PATH,
    "confusion_matrix_nyha_weighted_focal.pdf"
)