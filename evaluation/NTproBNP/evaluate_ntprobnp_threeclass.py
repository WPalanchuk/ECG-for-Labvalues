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
NICE_1LEAD_PATH = "PATH_TO_NICE_1LEAD_RUN_FOLDER"

NICE_12LEAD_PATH = "PATH_TO_NICE_12LEAD_RUN_FOLDER"

def evaluate_threeclass(path):
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
        display_labels=["<400", "400-2000", ">2000"],
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

print("1-lead")
evaluate_threeclass(NICE_1LEAD_PATH)

print("\n12-lead")
evaluate_threeclass(NICE_12LEAD_PATH)

make_confusion_matrix(
    NICE_1LEAD_PATH,
    "confusion_matrix_threeclass_1lead.pdf"
)

make_confusion_matrix(
    NICE_12LEAD_PATH,
    "confusion_matrix_threeclass_12lead.pdf"
)