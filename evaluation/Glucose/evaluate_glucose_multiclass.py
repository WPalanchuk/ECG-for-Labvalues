from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.utils import resample

# Replace with the path to your predictions file
pred_path = Path(
    "PATH_TO_PREDS_TEST.npz" 
)

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def bootstrap_macro_auroc(
    targs,
    probs,
    n_iterations=1000,
    alpha=0.95
):
    score_point = roc_auc_score(
        targs,
        probs,
        multi_class="ovr",
        average="macro"
    )

    scores = []

    for _ in range(n_iterations):
        sample_ids = resample(
            np.arange(len(targs)),
            n_samples=len(targs)
        )

        sample_targs = targs[sample_ids]
        sample_probs = probs[sample_ids]

        if len(np.unique(sample_targs)) < probs.shape[1]:
            continue

        score = roc_auc_score(
            sample_targs,
            sample_probs,
            multi_class="ovr",
            average="macro"
        )

        scores.append(score)

    scores = np.array(scores)
    score_diff = scores - score_point

    lower = score_point + np.percentile(
        score_diff,
        ((1 - alpha) / 2) * 100
    )

    upper = score_point + np.percentile(
        score_diff,
        (alpha + ((1 - alpha) / 2)) * 100
    )

    return score_point, lower, upper


data = np.load(pred_path)

preds = data["preds"]
targs = data["targs"]
ids = data["ids"]

print("Segments:", len(targs))
print("ECGs:", len(np.unique(ids)))

if targs.ndim > 1:
    if targs.shape[1] == 1:
        targs = targs.flatten()
    else:
        targs = np.argmax(targs, axis=1)

targs = targs.astype(int)

if (
    np.min(preds) < 0
    or np.max(preds) > 1
    or not np.allclose(
        np.sum(preds, axis=1),
        1,
        atol=1e-3
    )
):
    probs = softmax(preds)
else:
    probs = preds

y_pred = np.argmax(probs, axis=1)

print()
print("Target class distribution:")
classes, counts = np.unique(
    targs,
    return_counts=True
)

for c, n in zip(classes, counts):
    print(
        f"Class {c}: {n} "
        f"({100 * n / len(targs):.2f}%)"
    )

print()
print("Predicted class distribution:")
classes_pred, counts_pred = np.unique(
    y_pred,
    return_counts=True
)

for c, n in zip(classes_pred, counts_pred):
    print(
        f"Class {c}: {n} "
        f"({100 * n / len(targs):.2f}%)"
    )

auroc, auroc_low, auroc_high = bootstrap_macro_auroc(
    targs,
    probs
)

accuracy = accuracy_score(
    targs,
    y_pred
)

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

cm = confusion_matrix(
    targs,
    y_pred
)

print()
print("Classification metrics")
print("Macro AUROC:", round(auroc, 4))
print(
    "95% CI:",
    round(auroc_low, 4),
    "-",
    round(auroc_high, 4)
)
print("Accuracy:", round(accuracy, 4))
print("Sensitivity macro:", round(sensitivity, 4))
print("F1-score macro:", round(f1, 4))

print()
print("Confusion matrix:")
print(cm)

ConfusionMatrixDisplay.from_predictions(
    targs,
    y_pred,
    display_labels=[
        "<70",
        "70-180",
        ">180"
    ],
    cmap="Blues"
)

plt.xlabel("Predicted class")
plt.ylabel("Reference class")
plt.tight_layout()

plt.savefig(
    "confusion_matrix_glucose.pdf",
    bbox_inches="tight",
    pad_inches=0
)

plt.close()

print()
print("Saved confusion_matrix_glucose.pdf")