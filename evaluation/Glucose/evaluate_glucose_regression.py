from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
import pandas as pd

# Replace with the paths to your prediction files
reg_path = Path(
    "PATH_TO_REG_PREDS_TEST.npz"
)

l1_path = Path(
    "PATH_TO_L1_PREDS_TEST.npz"
)

huber_path = Path(
    "PATH_TO_HUBER_PREDS_TEST.npz"
)

gaussian_path = Path(
    "PATH_TO_GAUSSIAN_PREDS_TEST.npz"
)

clarke_path = Path(
    "PATH_TO_CLARKE_PREDS_TEST.npz"
)


def load_data(path, gaussian=False):
    data = np.load(path)

    preds = np.asarray(data["preds"])
    targs = np.asarray(data["targs"]).flatten()

    if gaussian and preds.ndim == 2 and preds.shape[1] > 1:
        preds = preds[:, 0]
    else:
        preds = preds.flatten()

    ids = data["ids"] if "ids" in data.files else None

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

    ax.set_title("Bland-Altman Plot - Glucose")
    ax.set_xlabel("Mean glucose [mg/dL]")
    ax.set_ylabel("Difference glucose [mg/dL]")

    plt.tight_layout()

    plt.savefig(
        "bland_altman_glucose.pdf",
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

print("Representative regression")
print("Segments:", len(targs))

if ids is not None:
    print("ECGs:", len(np.unique(ids)))
else:
    print("ECGs:", len(targs) // 4)

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
    "passing_bablok_glucose_nonnegative.csv"
)

print()
print("MAE")
print("MAE:", round(mae, 4), "mg/dL")
print(
    "95% CI:",
    round(mae_low, 4),
    "-",
    round(mae_high, 4),
    "mg/dL"
)

print()
print("Bland-Altman")
print("Bias:", round(bias, 4), "mg/dL")
print("Lower LoA:", round(loa_low, 4), "mg/dL")
print("Upper LoA:", round(loa_high, 4), "mg/dL")

print()
print("Passing-Bablok")
print("Included:", n_pb)
print("Excluded negative predictions:", n_negative)

experiments = [
    ("L1 loss", l1_path, False),
    ("Huber loss", huber_path, False),
    ("Gaussian negative log-likelihood", gaussian_path, True),
    ("Weighted Clarke error grid loss", clarke_path, False)
]

results = []

for name, path, gaussian in experiments:

    preds_config, targs_config, ids_config = load_data(
        path,
        gaussian=gaussian
    )

    mae_config, low_config, high_config = bootstrap_mae(
        targs_config,
        preds_config
    )

    if ids_config is not None:
        n_ecgs = len(np.unique(ids_config))
    else:
        n_ecgs = len(targs_config) // 4

    print()
    print(name)
    print("Segments:", len(targs_config))
    print("ECGs:", n_ecgs)

    print(
        "MAE:",
        round(mae_config, 4),
        "mg/dL"
    )

    print(
        "95% CI:",
        round(low_config, 4),
        "-",
        round(high_config, 4),
        "mg/dL"
    )

    results.append(
        (
            name,
            mae_config,
            low_config,
            high_config
        )
    )

for name, mae_config, low_config, high_config in results:
    print(
        f"{name} & "
        f"{mae_config:.2f} & "
        f"{low_config:.2f}--{high_config:.2f} \\\\"
    )