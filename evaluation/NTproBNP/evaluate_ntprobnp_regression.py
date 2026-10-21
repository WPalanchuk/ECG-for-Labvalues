from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
import pandas as pd

# Replace with the paths to your predictions files and scaler file
reg1_path = Path(
    "PATH_TO_1LEAD_PREDS_TEST.npz"
)

reg12_path = Path(
    "PATH_TO_12LEAD_PREDS_TEST.npz"
)

scaler_path = Path(
    "PATH_TO_SCALER.pkl"
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


def get_common_limits(preds1, targs1, preds12, targs12):
    means1 = (preds1 + targs1) / 2
    diff1 = preds1 - targs1

    means12 = (preds12 + targs12) / 2
    diff12 = preds12 - targs12

    all_means = np.concatenate([means1, means12])
    all_diffs = np.concatenate([diff1, diff12])

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

    filename = f"bland_altman_{name.replace('-', '_')}.pdf"

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


preds1, targs1, ids1 = load_data(reg1_path)
preds12, targs12, ids12 = load_data(reg12_path)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

preds1 = scaler.inverse_transform(
    preds1.reshape(-1, 1)
).flatten()

targs1 = scaler.inverse_transform(
    targs1.reshape(-1, 1)
).flatten()

preds12 = scaler.inverse_transform(
    preds12.reshape(-1, 1)
).flatten()

targs12 = scaler.inverse_transform(
    targs12.reshape(-1, 1)
).flatten()

print("1-lead segments:", len(targs1))
print("12-lead segments:", len(targs12))
print("Targets equal:", np.allclose(targs1, targs12))

mae1, mae1_low, mae1_high = bootstrap_mae(
    targs1,
    preds1
)

mae12, mae12_low, mae12_high = bootstrap_mae(
    targs12,
    preds12
)

xlim, ylim = get_common_limits(
    preds1,
    targs1,
    preds12,
    targs12
)

bias1, loa1_low, loa1_high = bland_altman(
    preds1,
    targs1,
    "1-lead",
    xlim,
    ylim
)

bias12, loa12_low, loa12_high = bland_altman(
    preds12,
    targs12,
    "12-lead",
    xlim,
    ylim
)

n_pb1, n_negative1 = export_passing_bablok(
    targs1,
    preds1,
    "passing_bablok_1lead_nonnegative.csv"
)

n_pb12, n_negative12 = export_passing_bablok(
    targs12,
    preds12,
    "passing_bablok_12lead_nonnegative.csv"
)

print()
print("1-lead MAE")
print("MAE:", round(mae1, 2), "pg/mL")
print(
    "95% CI:",
    round(mae1_low, 2),
    "-",
    round(mae1_high, 2),
    "pg/mL"
)

print()
print("12-lead MAE")
print("MAE:", round(mae12, 2), "pg/mL")
print(
    "95% CI:",
    round(mae12_low, 2),
    "-",
    round(mae12_high, 2),
    "pg/mL"
)

print()
print("1-lead Bland-Altman")
print("Bias:", round(bias1, 2), "pg/mL")
print("Lower LoA:", round(loa1_low, 2), "pg/mL")
print("Upper LoA:", round(loa1_high, 2), "pg/mL")

print()
print("12-lead Bland-Altman")
print("Bias:", round(bias12, 2), "pg/mL")
print("Lower LoA:", round(loa12_low, 2), "pg/mL")
print("Upper LoA:", round(loa12_high, 2), "pg/mL")

print()
print("Passing-Bablok")
print("1-lead included:", n_pb1)
print("1-lead excluded negative predictions:", n_negative1)
print("12-lead included:", n_pb12)
print("12-lead excluded negative predictions:", n_negative12)