import duckdb
import pandas as pd
import numpy as np
import os
import json
import pickle

from sklearn.preprocessing import StandardScaler, MinMaxScaler

TARGET_NAME = "Hemoglobin"
TARGET_ITEMID = 51222

FILTER_MIN = 0

median_BMI = 0
median_Height = 0
median_Weight = 0
deltaTVitals = 1
MAX_OMR_AGE_DAYS = 183

machine_measurements = "PATH_TO_MACHINE_MEASUREMENTS.csv"
labevents_csv = "PATH_TO_LABEVENTS.csv"
patients_table_path = "PATH_TO_PATIENTS.csv"
omr_table_path = "PATH_TO_OMR.csv"
record_list_path = "PATH_TO_RECORD_LIST.csv"
strat_fold_table = "PATH_TO_STRAT_FOLD.csv"
vitalsign_csv = "PATH_TO_VITALSIGN.csv"

output_path = "PATH_TO_OUTPUT_DIRECTORY"

results = []
connection = duckdb.connect()

print("Loading machine measurements table")
connection.execute(f"""
CREATE OR REPLACE TABLE ecg AS 
SELECT subject_id, study_id, CAST(ecg_time AS TIMESTAMP) AS ecg_time,
       rr_interval, p_onset, p_end, qrs_onset, qrs_end,
       t_end, p_axis, qrs_axis, t_axis
FROM read_csv_auto('{machine_measurements}')
WHERE ecg_time IS NOT NULL
""")

print("Loading patients table")
connection.execute(f"""
CREATE OR REPLACE TABLE patients AS
SELECT subject_id,
       CASE WHEN gender = 'M' THEN 1 ELSE 0 END AS gender,
       anchor_age,
       anchor_year,
       anchor_year_group
FROM read_csv_auto('{patients_table_path}')
""")

print("Loading record table")
connection.execute(f"""
CREATE OR REPLACE TABLE record AS 
SELECT study_id, path, row_number() OVER () - 1 AS data
FROM read_csv_auto('{record_list_path}')
""")

print("Loading strat_fold table")
connection.execute(f"""
CREATE OR REPLACE TABLE strat_fold AS
SELECT *
FROM read_csv_auto('{strat_fold_table}')
""")

print("Loading omr table")
connection.execute(f"""
CREATE OR REPLACE TABLE omr AS
SELECT subject_id,
       CAST(chartdate AS TIMESTAMP) AS chartdate,
       seq_num,
       result_name,
       TRY_CAST(result_value AS DOUBLE) AS result_value
FROM read_csv_auto('{omr_table_path}')
WHERE (result_name = 'BMI (kg/m2)'
    OR result_name = 'Height (Inches)'
    OR result_name = 'Weight (Lbs)')
  AND TRY_CAST(result_value AS DOUBLE) IS NOT NULL
  AND result_name IS NOT NULL
""")

print("Loading ED vitalsign table")
connection.execute(f"""
CREATE OR REPLACE TABLE vitals AS
SELECT
    subject_id,
    stay_id,
    CAST(charttime AS TIMESTAMP) AS charttime,
    TRY_CAST(temperature AS DOUBLE) AS temperature,
    TRY_CAST(heartrate AS DOUBLE) AS heartrate,
    TRY_CAST(resprate AS DOUBLE) AS resprate,
    TRY_CAST(o2sat AS DOUBLE) AS o2sat,
    TRY_CAST(sbp AS DOUBLE) AS sbp,
    TRY_CAST(dbp AS DOUBLE) AS dbp
FROM read_csv_auto('{vitalsign_csv}')
WHERE charttime IS NOT NULL
""")

print(f"Loading {TARGET_NAME} labevents table")
connection.execute(f"""
CREATE OR REPLACE TABLE lab AS
SELECT labevent_id,
       subject_id,
       CAST(charttime AS TIMESTAMP) AS charttime,
       valuenum,
       itemid,
       valueuom
FROM read_csv_auto('{labevents_csv}')
WHERE itemid = {TARGET_ITEMID}
  AND charttime IS NOT NULL
  AND valuenum IS NOT NULL
  AND valueuom IS NOT NULL
  AND valuenum > 0
""")

for deltaT in [4 / 60, 3 / 60, 2 / 60, 1 / 60]:
    print(f"\nMatching {TARGET_NAME} to ECG, deltaT={deltaT}")

    connection.execute(f"""
    CREATE OR REPLACE TABLE matched AS
    SELECT l.labevent_id,
           l.subject_id,
           l.charttime AS lab_time,
           l.valuenum AS lab_value,
           l.itemid,
           e.study_id,
           e.ecg_time,
           DATE_DIFF('minute', l.charttime, e.ecg_time) AS timedifference_minutes,
           e.rr_interval,
           e.p_onset,
           e.p_end,
           e.qrs_onset,
           e.qrs_end,
           e.t_end,
           e.p_axis,
           e.qrs_axis,
           e.t_axis
    FROM lab l
    JOIN ecg e ON l.subject_id = e.subject_id
    WHERE ABS(DATE_DIFF('minute', l.charttime, e.ecg_time)) <= {deltaT * 60}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY e.study_id
        ORDER BY ABS(DATE_DIFF('minute', l.charttime, e.ecg_time)) ASC,
                 l.charttime ASC,
                 l.labevent_id ASC
    ) = 1
    """)

    print("Removing duplicate lab measurements so it is 1-to-1")
    connection.execute("""
    CREATE OR REPLACE TABLE matched AS
    SELECT *
    FROM matched
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY labevent_id
        ORDER BY ABS(timedifference_minutes) ASC,
                 ecg_time ASC
    ) = 1
    """)

    print("Creating matched_with_patients table")
    connection.execute(f"""
    CREATE OR REPLACE TABLE matched_with_patients AS
    SELECT b.subject_id,
           b.study_id AS ecg_id,
           b.labevent_id AS {TARGET_NAME}_id,
           p.anchor_age + (EXTRACT(YEAR FROM b.ecg_time) - p.anchor_year) AS age,
           p.gender,
           b.lab_value AS {TARGET_NAME}_value,
           b.ecg_time,
           b.lab_time AS {TARGET_NAME}_time,
           b.timedifference_minutes,
           b.rr_interval,
           b.p_onset,
           b.p_end,
           b.qrs_onset,
           b.qrs_end,
           b.t_end,
           b.p_axis,
           b.qrs_axis,
           b.t_axis
    FROM matched b
    LEFT JOIN patients p ON b.subject_id = p.subject_id
    """)

    print("Creating omr_pivot table")
    connection.execute(f"""
    CREATE OR REPLACE TABLE omr_pivot AS
    PIVOT(
        SELECT m.ecg_id,
               o.result_name AS omr_name,
               o.result_value AS omr_value
        FROM matched_with_patients m
        JOIN omr o
          ON m.subject_id = o.subject_id
         AND ABS(DATE_DIFF('day', m.ecg_time, o.chartdate)) <= {MAX_OMR_AGE_DAYS}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY m.ecg_id, o.result_name
            ORDER BY ABS(DATE_DIFF('day', m.ecg_time, o.chartdate))
        ) = 1
    )
    ON omr_name
    USING MAX(omr_value)
    GROUP BY ecg_id
    """)
    
    print("Creating vitals_matched table")
    connection.execute(f"""
    CREATE OR REPLACE TABLE vitals_matched AS
    SELECT
        m.ecg_id,
        v.charttime AS vitals_time,
        DATE_DIFF('minute', v.charttime, m.ecg_time) AS vitals_timedifference_minutes,
        v.temperature,
        v.heartrate,
        v.resprate,
        v.o2sat,
        v.sbp,
        v.dbp
    FROM matched_with_patients m
    JOIN vitals v
    ON m.subject_id = v.subject_id
    AND ABS(DATE_DIFF('minute', v.charttime, m.ecg_time)) <= {deltaTVitals * 60}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY m.ecg_id
        ORDER BY ABS(DATE_DIFF('minute', v.charttime, m.ecg_time)) ASC,
                v.charttime ASC
    ) = 1
    """)

    print("Creating final table")
    connection.execute("""
    CREATE OR REPLACE TABLE final AS
    SELECT m.*,
            o."BMI (kg/m2)" AS bmi,
            o."Height (Inches)" AS height_inches,
            o."Weight (Lbs)" AS weight_lbs,
            v.vitals_time,
            v.vitals_timedifference_minutes,
            v.temperature,
            v.heartrate,
            v.resprate,
            v.o2sat,
            v.sbp,
            v.dbp,
            CASE WHEN v.temperature IS NULL THEN 1 ELSE 0 END AS temperature_missing,
            CASE WHEN v.heartrate IS NULL THEN 1 ELSE 0 END AS heartrate_missing,
            CASE WHEN v.resprate IS NULL THEN 1 ELSE 0 END AS resprate_missing,
            CASE WHEN v.o2sat IS NULL THEN 1 ELSE 0 END AS o2sat_missing,
            CASE WHEN v.sbp IS NULL THEN 1 ELSE 0 END AS sbp_missing,
            CASE WHEN v.dbp IS NULL THEN 1 ELSE 0 END AS dbp_missing,
            CASE WHEN o."BMI (kg/m2)" IS NULL THEN 1 ELSE 0 END AS bmi_missing,
            CASE WHEN o."Weight (Lbs)" IS NULL THEN 1 ELSE 0 END AS weight_missing,
            CASE WHEN o."Height (Inches)" IS NULL THEN 1 ELSE 0 END AS height_missing,
            r.path AS ecg_path,
            sf.strat_fold,
            r.data,
            SIN(2 * PI() * (EXTRACT(HOUR FROM ecg_time) + EXTRACT(MINUTE FROM ecg_time) / 60.0) / 24.0) AS ecg_tod_sin,
            COS(2 * PI() * (EXTRACT(HOUR FROM ecg_time) + EXTRACT(MINUTE FROM ecg_time) / 60.0) / 24.0) AS ecg_tod_cos
    FROM matched_with_patients m
    LEFT JOIN omr_pivot o ON m.ecg_id = o.ecg_id
    LEFT JOIN record r ON m.ecg_id = r.study_id
    LEFT JOIN strat_fold sf ON m.subject_id = sf.subject_id
    LEFT JOIN vitals_matched v ON m.ecg_id = v.ecg_id
    """)

    print("Dataset statistics before transform")
    stats = connection.execute(f"""
    SELECT
        COUNT(*) AS rows,
        COUNT(DISTINCT subject_id) AS subjects,
        COUNT(DISTINCT ecg_id) AS ecgs,
        COUNT(DISTINCT {TARGET_NAME}_id) AS lab_measurements,
        MIN({TARGET_NAME}_value) AS min_value,
        MEDIAN({TARGET_NAME}_value) AS median_value,
        AVG({TARGET_NAME}_value) AS mean_value,
        MAX({TARGET_NAME}_value) AS max_value
    FROM final
    """).df()
    print(stats)
    
    print("Getting medians of metadata features from train fold")
    median_BMI = connection.execute(
        "SELECT median(bmi) FROM final WHERE strat_fold = 0"
    ).fetchone()[0]
    median_Height = connection.execute(
        "SELECT median(height_inches) FROM final WHERE strat_fold = 0"
    ).fetchone()[0]
    median_Weight = connection.execute(
        "SELECT median(weight_lbs) FROM final WHERE strat_fold = 0"
    ).fetchone()[0]

    print("Adding metadata medians")
    connection.execute(f"UPDATE final SET bmi = {median_BMI} WHERE bmi IS NULL")
    connection.execute(f"UPDATE final SET height_inches = {median_Height} WHERE height_inches IS NULL")
    connection.execute(f"UPDATE final SET weight_lbs = {median_Weight} WHERE weight_lbs IS NULL")
    
    vital_cols = ["temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]

    vital_medians = {}
    for col in vital_cols:
        vital_medians[col] = connection.execute(
            f"SELECT median({col}) FROM final WHERE strat_fold = 0"
        ).fetchone()[0]

    for col, med in vital_medians.items():
        if med is not None:
            connection.execute(f"UPDATE final SET {col} = {med} WHERE {col} IS NULL")
    
    feature_cols = [
    "age", "gender",
    "rr_interval", "p_onset", "p_end",
    "qrs_onset", "qrs_end", "t_end",
    "p_axis", "qrs_axis", "t_axis",
    "bmi", "height_inches", "weight_lbs",
    "bmi_missing", "weight_missing", "height_missing",
    "ecg_tod_sin", "ecg_tod_cos",
    "timedifference_minutes",
    "timedifference_hours",
    "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp",
    "temperature_missing", "heartrate_missing", "resprate_missing",
    "o2sat_missing", "sbp_missing", "dbp_missing",
    ]

    df_full = connection.execute("SELECT * FROM final").df()

    df_full["timedifference_hours"] = df_full["timedifference_minutes"] / 60.0
    df_full[feature_cols] = df_full[feature_cols].astype(float)
    df_full["features"] = df_full[feature_cols].values.tolist()

    raw_col = f"{TARGET_NAME}_value"
    original_col = f"{TARGET_NAME}_value_original"

    df_full[original_col] = df_full[raw_col].astype(float)
    df_full = df_full[df_full[original_col] > FILTER_MIN].copy()
    
    print(f"Remaining samples: {len(df_full)}")
    print(df_full[original_col].describe())

    train_mask = df_full["strat_fold"] == 0
    train_values = df_full.loc[train_mask, [original_col]].values

    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    standard_scaler.fit(train_values)
    minmax_scaler.fit(train_values)

    scalers = {
        "standard": {
            "scaler": standard_scaler,
            "scaled_col": f"{TARGET_NAME}_value_standard_scaled",
            "transform": "filter_standard",
        },
        "minmax": {
            "scaler": minmax_scaler,
            "scaled_col": f"{TARGET_NAME}_value_minmax_scaled",
            "transform": "filter_minmax",
        },
    }

    for scaler_name, scaler_cfg in scalers.items():
        scaler = scaler_cfg["scaler"]
        scaled_col = scaler_cfg["scaled_col"]

        df_full[scaled_col] = scaler.transform(df_full[[original_col]].values).flatten()

    base_cols_to_keep = [
        "data",
        "subject_id",
        "ecg_id",
        f"{TARGET_NAME}_id",
        "ecg_time",
        f"{TARGET_NAME}_time",
        "timedifference_minutes",
        "timedifference_hours",
        "strat_fold",
        "ecg_path",
        "features",
        original_col,
    ]

    folder_name = f"{int(deltaT)}h" if deltaT >= 1 else f"{round(deltaT * 60)}min"

    for scaler_name, scaler_cfg in scalers.items():
        scaler = scaler_cfg["scaler"]
        scaled_col = scaler_cfg["scaled_col"]

        df_model = df_full[base_cols_to_keep + [scaled_col]].copy()
        df_model["label"] = df_model[scaled_col].apply(lambda x: [float(x)])

        cols_to_keep = base_cols_to_keep + ["label"]
        df_model = df_model[cols_to_keep]

        out_dir = f"{output_path}/{scaler_name}/{folder_name}"
        os.makedirs(out_dir, exist_ok=True)

        df_model.to_pickle(f"{out_dir}/mds_icu_preprocessed2_ecg.pkl")
        df_model.to_parquet(f"{out_dir}/mds_icu_preprocessed2_ecg.parquet")
        np.save(f"{out_dir}/lbl_itos.npy", np.array([TARGET_NAME]))

        scaler_info = {
            "target_name": TARGET_NAME,
            "itemid": TARGET_ITEMID,
            "FILTER_MIN": FILTER_MIN,
            "transform": scaler_cfg["transform"],
            "label_column": "label",
            "label_scale": scaler_name,
            "original_column": original_col,
        }

        if scaler_name == "standard":
            scaler_info.update({
                "mean_": standard_scaler.mean_.tolist(),
                "var_": standard_scaler.var_.tolist(),
                "scale_": standard_scaler.scale_.tolist(),
            })

        if scaler_name == "minmax":
            scaler_info.update({
                "data_min_": minmax_scaler.data_min_.tolist(),
                "data_max_": minmax_scaler.data_max_.tolist(),
                "data_range_": minmax_scaler.data_range_.tolist(),
                "scale_": minmax_scaler.scale_.tolist(),
                "min_": minmax_scaler.min_.tolist(),
                "feature_range": minmax_scaler.feature_range,
            })

        with open(f"{out_dir}/target_scaler.json", "w") as f:
            json.dump(scaler_info, f, indent=2)

        with open(f"{out_dir}/target_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        if deltaT >= 1:
            df_model.to_csv(
                f"{out_dir}/{TARGET_NAME}_ecg_pairs_{int(deltaT)}h.csv",
                index=False,
            )
        else:
            df_model.to_csv(
                f"{out_dir}/{TARGET_NAME}_ecg_pairs_{int(deltaT * 60)}min.csv",
                index=False,
            )

        print(f"\nSaved {scaler_name} dataset to:")
        print(out_dir)
        print("Label statistics:")
        print(df_model["label"].apply(lambda x: x[0]).describe())

    print(f"Saving result data for {deltaT}")

    df = pd.DataFrame([{
        "deltaT_hours": deltaT,
        "matched_rows_after_filter": len(df_full),
        "subjects_after_filter": df_full["subject_id"].nunique(),
        "ecgs_after_filter": df_full["ecg_id"].nunique(),
        "lab_measurements_after_filter": df_full[f"{TARGET_NAME}_id"].nunique(),

        "min_value_after_filter": df_full[original_col].min(),
        "median_value_after_filter": df_full[original_col].median(),
        "mean_value_after_filter": df_full[original_col].mean(),
        "max_value_after_filter": df_full[original_col].max(),

        "avg_timediff": df_full["timedifference_minutes"].mean(),
        "min_timediff": df_full["timedifference_minutes"].min(),
        "max_timediff": df_full["timedifference_minutes"].max(),

        "train_percentage": round(100.0 * (df_full["strat_fold"] == 0).sum() / len(df_full), 2),
        "validate_percentage": round(100.0 * (df_full["strat_fold"] == 1).sum() / len(df_full), 2),
        "test_percentage": round(100.0 * (df_full["strat_fold"] == 2).sum() / len(df_full), 2),

        "bmi_missing_count": int((df_full["bmi_missing"] == 1).sum()),
        "weight_missing": int((df_full["weight_missing"] == 1).sum()),
        "height_missing": int((df_full["height_missing"] == 1).sum()),

        "temperature_missing": int((df_full["temperature_missing"] == 1).sum()),
        "heartrate_missing": int((df_full["heartrate_missing"] == 1).sum()),
        "resprate_missing": int((df_full["resprate_missing"] == 1).sum()),
        "o2sat_missing": int((df_full["o2sat_missing"] == 1).sum()),
        "sbp_missing": int((df_full["sbp_missing"] == 1).sum()),
        "dbp_missing": int((df_full["dbp_missing"] == 1).sum()),

        "min_data": df_full["data"].min(),
        "max_data": df_full["data"].max(),
    }])

    results.append(df)

result_table = pd.concat(results, ignore_index=True)
result_table = result_table.sort_values("deltaT_hours", ascending=False)

print("\nResult table:")
print(result_table)