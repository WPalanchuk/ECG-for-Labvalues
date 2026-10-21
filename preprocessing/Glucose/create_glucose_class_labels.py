import pandas as pd
import numpy as np
from pathlib import Path

input_path = Path("PATH_TO_PREPROCESSED_GLUCOSE.pkl")
out_dir = Path("PATH_TO_OUTPUT_DIRECTORY")
output_path = out_dir / "mds_icu_preprocessed2_ecg"

df = pd.read_pickle(input_path)

def glucose_to_class(x):
    glucose = x[0]

    if glucose < 70:
        return [0]

    if glucose <= 180:
        return [1]

    return [2]

df["label"] = df["label"].apply(glucose_to_class)

df.to_pickle(str(output_path) + ".pkl")
df.to_parquet(str(output_path) + ".parquet")

print(df["label"].apply(lambda x: x[0]).value_counts().sort_index())
print(df["strat_fold"].value_counts().sort_index())

np.save(out_dir / "lbl_itos.npy", np.array(["hypo", "normal", "hyper"]))