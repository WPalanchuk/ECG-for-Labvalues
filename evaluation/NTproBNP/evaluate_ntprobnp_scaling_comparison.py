from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
import pandas as pd

# Replace with the paths to your predictions files and scaler files
minmax_path = Path(
    "PATH_TO_MINMAX_PREDS_TEST.npz"
)

standard_path = Path(
    "PATH_TO_STANDARD_PREDS_TEST.npz"
)

minmax_scaler_path = Path(
    "PATH_TO_MINMAX_SCALER.pkl"
)

standard_scaler_path = Path(
    "PATH_TO_STANDARD_SCALER.pkl"
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


def get_common_limits(preds1, targs1, preds2, targs2):
    means1 = (preds1 + targs1) / 2
    diff1 = preds1 - targs1

    means2 = (preds2 + targs2) / 2
    diff2 = preds2 - targs2

    all_means = np.concatenate([means1, means2])
    all_diffs = np.concatenate([diff1, diff2])

    x_margin = 0.05 * (all_means.max() - all_means.min())
    y_margin = 0.05 * (all_diffs.max() - all_diffs.min())

    xlim = (
        all_means.min() - x_margin,
        all_means.max() + x_margin
    )

    ylim = (
        all_diffs.min() - y_margin,
        all_diffs.max() + y_margin
    )

    return xlim, ylim


def bland_altman(preds, targs, name, xlim, ylim):
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

    ax.set_title(f"Bland-Altman Plot - {name}")
    ax.set_xlabel("Mean NT-proBNP [pg/mL]")
    ax.set_ylabel("Difference NT-proBNP [pg/mL]")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plt.tight_layout()

    filename = f"bland_altman_{name.replace(' ', '_').lower()}.pdf"

    plt.savefig(
        filename,
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


preds_minmax, targs_minmax, ids_minmax = load_data(minmax_path)
preds_standard, targs_standard, ids_standard = load_data(standard_path)

with open(minmax_scaler_path, "rb") as f:
    minmax_scaler = pickle.load(f)

with open(standard_scaler_path, "rb") as f:
    standard_scaler = pickle.load(f)

preds_minmax = minmax_scaler.inverse_transform(
    preds_minmax.reshape(-1, 1)
).flatten()

targs_minmax = minmax_scaler.inverse_transform(
    targs_minmax.reshape(-1, 1)
).flatten()

preds_standard = standard_scaler.inverse_transform(
    preds_standard.reshape(-1, 1)
).flatten()

targs_standard = standard_scaler.inverse_transform(
    targs_standard.reshape(-1, 1)
).flatten()

preds_minmax = np.expm1(preds_minmax.astype(np.float64))
targs_minmax = np.expm1(targs_minmax.astype(np.float64))

preds_standard = np.expm1(preds_standard.astype(np.float64))
targs_standard = np.expm1(targs_standard.astype(np.float64))

print("MinMax segments:", len(targs_minmax))
print("Standard segments:", len(targs_standard))
print("Targets equal:", np.allclose(targs_minmax, targs_standard))

mae_minmax, mae_minmax_low, mae_minmax_high = bootstrap_mae(
    targs_minmax,
    preds_minmax
)

mae_standard, mae_standard_low, mae_standard_high = bootstrap_mae(
    targs_standard,
    preds_standard
)

xlim, ylim = get_common_limits(
    preds_minmax,
    targs_minmax,
    preds_standard,
    targs_standard
)

bias_minmax, loa_minmax_low, loa_minmax_high = bland_altman(
    preds_minmax,
    targs_minmax,
    "MinMax",
    xlim,
    ylim
)

bias_standard, loa_standard_low, loa_standard_high = bland_altman(
    preds_standard,
    targs_standard,
    "Standard",
    xlim,
    ylim
)

n_pb_minmax, n_negative_minmax = export_passing_bablok(
    targs_minmax,
    preds_minmax,
    "passing_bablok_minmax_nonnegative.csv"
)

n_pb_standard, n_negative_standard = export_passing_bablok(
    targs_standard,
    preds_standard,
    "passing_bablok_standard_nonnegative.csv"
)

print()
print("MinMax MAE")
print("MAE:", round(mae_minmax, 2), "pg/mL")
print(
    "95% CI:",
    round(mae_minmax_low, 2),
    "-",
    round(mae_minmax_high, 2),
    "pg/mL"
)

print()
print("Standard MAE")
print("MAE:", round(mae_standard, 2), "pg/mL")
print(
    "95% CI:",
    round(mae_standard_low, 2),
    "-",
    round(mae_standard_high, 2),
    "pg/mL"
)

print()
print("MinMax Bland-Altman")
print("Bias:", round(bias_minmax, 2), "pg/mL")
print("Lower LoA:", round(loa_minmax_low, 2), "pg/mL")
print("Upper LoA:", round(loa_minmax_high, 2), "pg/mL")

print()
print("Standard Bland-Altman")
print("Bias:", round(bias_standard, 2), "pg/mL")
print("Lower LoA:", round(loa_standard_low, 2), "pg/mL")
print("Upper LoA:", round(loa_standard_high, 2), "pg/mL")

print()
print("Passing-Bablok")
print("MinMax included:", n_pb_minmax)
print("MinMax excluded negative predictions:", n_negative_minmax)
print("Standard included:", n_pb_standard)
print("Standard excluded negative predictions:", n_negative_standard)