from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
import pandas as pd

# Replace with the path to your predictions file
reg_path = Path(
    "PATH_TO_PREDS_TEST.npz"
)

def load_data(path):
    data = np.load(path)

    preds = data["preds"].flatten()
    targs = data["targs"].flatten()
    ids = data["ids"]

    return preds, targs, ids

def bootstrap_mae(targs, preds, n_iterations=1000, alpha=0.95):
    score_point = mean_absolute_error(targs, preds)
    scores = []

    for _ in range(n_iterations):
        sample_ids = resample(
            np.arange(len(targs)),
            n_samples=len(targs)
        )

        scores.append(
            mean_absolute_error(
                targs[sample_ids],
                preds[sample_ids]
            )
        )

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

def bland_altman(preds, targs):
    differences = preds - targs

    bias = np.mean(differences)
    sd = np.std(differences)

    lower_loa = bias - 1.96 * sd
    upper_loa = bias + 1.96 * sd

    fig, ax = plt.subplots(figsize=(7, 5))

    sm.graphics.mean_diff_plot(
        preds,
        targs,
        sd_limit=1.96,
        ax=ax
    )

    ax.set_title("Bland-Altman Plot - Potassium")
    ax.set_xlabel("Mean potassium [mEq/L]")
    ax.set_ylabel("Difference potassium [mEq/L]")

    plt.tight_layout()

    plt.savefig(
        "bland_altman_potassium.pdf",
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    return bias, lower_loa, upper_loa

def export_passing_bablok(targs, preds, output_path):
    mask = preds >= 0

    pd.DataFrame({
        "reference": targs[mask],
        "prediction": preds[mask]
    }).to_csv(
        output_path,
        index=False
    )

    return mask.sum(), np.sum(~mask)

preds, targs, ids = load_data(reg_path)

print("Segments:", len(targs))
print("ECGs:", len(np.unique(ids)))

mae, mae_low, mae_high = bootstrap_mae(
    targs,
    preds
)

bias, loa_low, loa_high = bland_altman(
    preds,
    targs
)

n_pb, n_negative = export_passing_bablok(
    targs,
    preds,
    "passing_bablok_potassium_nonnegative.csv"
)

print()
print("Potassium MAE")
print("MAE:", round(mae, 4), "mEq/L")
print(
    "95% CI:",
    round(mae_low, 4),
    "-",
    round(mae_high, 4),
    "mEq/L"
)

print()
print("Bland-Altman")
print("Bias:", round(bias, 4), "mEq/L")
print("Lower LoA:", round(loa_low, 4), "mEq/L")
print("Upper LoA:", round(loa_high, 4), "mEq/L")

print()
print("Passing-Bablok")
print("Included:", n_pb)
print("Excluded negative predictions:", n_negative)