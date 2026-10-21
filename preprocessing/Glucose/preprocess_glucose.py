import duckdb
import pandas as pd
import numpy as np
import os

glucose_itemid_dict = {'Blood_Chemistry_1': 50931}

median_BMI = 0
median_Height = 0
median_Weight = 0
MAX_OMR_AGE_DAYS = 183

machine_measurements = "PATH_TO_MACHINE_MEASUREMENTS.csv"
labevents_csv = "PATH_TO_LABEVENTS.csv"
patients_table_path = "PATH_TO_PATIENTS.csv"
omr_table_path = "PATH_TO_OMR.csv"
record_list_path = "PATH_TO_RECORD_LIST.csv"
strat_fold_table = "PATH_TO_STRAT_FOLD.csv"

output_path = "PATH_TO_OUTPUT_DIRECTORY"

results = []

connection = duckdb.connect()

print('Loading machine measurements table')
connection.execute(f"""
                   CREATE OR REPLACE TABLE ecg AS 
                   SELECT subject_id, study_id, CAST(ecg_time AS TIMESTAMP) as ecg_time,
                   rr_interval, p_onset, p_end, qrs_onset, qrs_end, t_end, p_axis, qrs_axis, t_axis
                   FROM read_csv_auto('{machine_measurements}')
                   WHERE ecg_time IS NOT NULL
""")
print('Loading machine measurements table completed')

print('Loading patients table')
connection.execute(f"""
                   CREATE OR REPLACE TABLE patients AS
                   SELECT subject_id, CASE WHEN gender = 'M' THEN 1 ELSE 0 END AS gender, anchor_age, anchor_year, anchor_year_group
                   FROM read_csv_auto('{patients_table_path}')
""")
print('Loading patients table completed')

print('Loading record table')
connection.execute(f"""
                   CREATE OR REPLACE TABLE record AS 
                   SELECT study_id, path, row_number() OVER () - 1 AS data
                   FROM read_csv_auto('{record_list_path}')
                   """)
print('Loading record table completed')

print('Loading strat_fold table')
connection.execute(f"""
                   CREATE OR REPLACE TABLE strat_fold AS
                   SELECT *
                   FROM read_csv_auto('{strat_fold_table}')
                   """)
print('Loading strat_fold table completed')

print('Loading omr table')
connection.execute(f"""
                   CREATE OR REPLACE TABLE omr AS
                   SELECT subject_id, CAST(chartdate AS TIMESTAMP) AS chartdate, seq_num, result_name, TRY_CAST(result_value AS DOUBLE) AS result_value
                   FROM read_csv_auto('{omr_table_path}')
                   WHERE (result_name = 'BMI (kg/m2)' OR result_name = 'Height (Inches)' OR result_name = 'Weight (Lbs)') AND TRY_CAST(result_value AS DOUBLE) IS NOT NULL AND result_name IS NOT NULL
                   """)
print('Loading omr table completed')

print('Loading labevents table')
connection.execute(f"""
                   CREATE OR REPLACE TABLE glucose AS
                   SELECT labevent_id, subject_id, CAST(charttime AS TIMESTAMP) AS charttime, valuenum, itemid, valueuom
                   FROM read_csv_auto('{labevents_csv}')
                   WHERE itemid = {glucose_itemid_dict['Blood_Chemistry_1']} AND charttime IS NOT NULL AND valuenum IS NOT NULL AND valueuom IS NOT NULL AND valuenum BETWEEN 20 AND 500
                   """)
print('Loading labevents table completed')

for deltaT in [5/60]:
    print('Matching glucose to ecg')
    connection.execute(f"""
                       CREATE OR REPLACE TABLE matched AS
                       SELECT g.labevent_id, g.subject_id, g.charttime AS glucose_time, g.valuenum, g.itemid, 
                       e.study_id, e.ecg_time, DATE_DIFF('hour', g.charttime, e.ecg_time) AS timedifference_hours, DATE_DIFF('minute', g.charttime, e.ecg_time) % 60 AS timedifference_minutes,
                       e.rr_interval, e.p_onset, e.p_end, e.qrs_onset, e.qrs_end, e.t_end, e.p_axis, e.qrs_axis, e.t_axis
                       FROM glucose g
                       JOIN ecg e ON g.subject_id = e.subject_id
                       WHERE ABS(DATE_DIFF('minute', g.charttime, e.ecg_time)) <= {deltaT * 60}
                       QUALIFY ROW_NUMBER() OVER (
                           PARTITION BY e.study_id
                           ORDER BY ABS(DATE_DIFF('minute', g.charttime, e.ecg_time)) ASC, g.charttime ASC, g.labevent_id ASC
                           ) = 1
                       """)
    print('Matching glucose to ecg completed')
    
    print('Removing duplicate glucose measurements so its 1to1')
    connection.execute(f"""
                       CREATE OR REPLACE TABLE matched AS
                        SELECT *
                        FROM matched
                        QUALIFY ROW_NUMBER() OVER (
                            PARTITION BY labevent_id
                            ORDER BY ABS(timedifference_minutes) ASC, ecg_time ASC
                            ) = 1
                       """)
    print('Finished removing duplicate glucose measurements')
    
    print('Creating matched_with_patients table with patient data')
    connection.execute(f"""
                       CREATE OR REPLACE TABLE matched_with_patients AS
                       SELECT b.subject_id, b.study_id AS ecg_id, labevent_id AS glucose_id, p.anchor_age + (EXTRACT(YEAR FROM b.ecg_time) - p.anchor_year) AS age,
                       p.gender, b.valuenum AS glucose_value, ecg_time, b.glucose_time, b.timedifference_minutes, b.timedifference_hours,
                       b.rr_interval, b.p_onset, b.p_end, b.qrs_onset, b.qrs_end, b.t_end, b.p_axis, b.qrs_axis, b.t_axis
                       FROM matched b
                       LEFT JOIN patients p ON b.subject_id = p.subject_id
                       """)
    print('Creating matched_with_patients table with patient data completed')

    print('Creating omr_pivot table')
    connection.execute(f"""
                       CREATE OR REPLACE TABLE omr_pivot AS
                       PIVOT(
                           SELECT m.ecg_id, o.result_name AS omr_name, o.result_value AS omr_value
                           FROM matched_with_patients m
                           JOIN omr o ON m.subject_id = o.subject_id AND ABS(DATE_DIFF('day', m.ecg_time, o.chartdate)) <= {MAX_OMR_AGE_DAYS}
                           QUALIFY ROW_NUMBER() OVER (
                               PARTITION BY m.ecg_id, o.result_name
                               ORDER BY ABS(DATE_DIFF('day', m.ecg_time, o.chartdate))
                               ) = 1
                       )
                       ON omr_name
                       USING MAX(omr_value)
                       GROUP BY ecg_id
                       """)
    print('Creating omr_pivot table completed')
    
    print('Creating final table')
    connection.execute(f"""
                       CREATE OR REPLACE TABLE final AS
                       SELECT m.*, o."BMI (kg/m2)" AS bmi, o."Height (Inches)" AS height_inches, o."Weight (Lbs)" AS weight_lbs, 
                       CASE WHEN o."BMI (kg/m2)" IS NULL THEN 1 ELSE 0 END AS bmi_missing, 
                       CASE WHEN o."Weight (Lbs)" IS NULL THEN 1 ELSE 0 END AS weight_missing, 
                       CASE WHEN o."Height (Inches)" IS NULL THEN 1 ELSE 0 END AS height_missing, 
                       r.path AS ecg_path, sf.strat_fold, r.data,
                       SIN(2 * PI() * (EXTRACT(HOUR FROM ecg_time) + EXTRACT(MINUTE FROM ecg_time) / 60.0) / 24.0) AS ecg_tod_sin, COS(2 * PI() * (EXTRACT(HOUR FROM ecg_time) + EXTRACT(MINUTE FROM ecg_time) / 60.0) / 24.0) AS ecg_tod_cos
                       FROM matched_with_patients m
                       LEFT JOIN omr_pivot o ON m.ecg_id = o.ecg_id
                       LEFT JOIN record r ON m.ecg_id = r.study_id
                       LEFT JOIN strat_fold sf ON m.subject_id = sf.subject_id
                       """)
    print('Creating final table completed')
    
    print('Getting medians of features')
    median_BMI = connection.execute(f"""
                                    SELECT median(bmi)
                                    FROM final
                                    WHERE strat_fold = 0
                                    """).fetchone()[0]
    median_Height = connection.execute(f"""
                                    SELECT median(height_inches)
                                    FROM final
                                    WHERE strat_fold = 0
                                    """).fetchone()[0]
    median_Weight = connection.execute(f"""
                                    SELECT median(weight_lbs)
                                    FROM final
                                    WHERE strat_fold = 0
                                    """).fetchone()[0]
    print('Getting medians of features completed')
    
    print('Adding median')
    connection.execute(f"""
                       UPDATE final
                       SET bmi = {median_BMI}
                       WHERE bmi IS NULL
                       """)
    connection.execute(f"""
                       UPDATE final
                       SET height_inches = {median_Height}
                       WHERE height_inches IS NULL
                       """)
    connection.execute(f"""
                       UPDATE final
                       SET weight_lbs = {median_Weight}
                       WHERE weight_lbs IS NULL
                       """)
    print('Adding median completed')
    
    feature_cols = [
    "age",
    "gender",
    "rr_interval",
    "p_onset",
    "p_end",
    "qrs_onset",
    "qrs_end",
    "t_end",
    "p_axis",
    "qrs_axis",
    "t_axis",
    "bmi",
    "height_inches",
    "weight_lbs",
    "bmi_missing",
    "weight_missing",
    "height_missing",
    "ecg_tod_sin",
    "ecg_tod_cos",
    "timedifference_hours",
    "timedifference_minutes",
    ]
    
    df_full = connection.execute("SELECT * FROM final").df()

    df_full[feature_cols] = df_full[feature_cols].astype(float)
    df_full["features"] = df_full[feature_cols].values.tolist()
    df_full["label"] = df_full["glucose_value"].astype(float).apply(lambda x: [x])
    
    cols_to_keep = [
    "data",
    "subject_id",
    "ecg_id",
    "glucose_id",
    "ecg_time",
    "glucose_time",
    "timedifference_hours",
    "timedifference_minutes",
    "strat_fold",
    "ecg_path",
    "features",
    "label"
    ]
    
    df_model = df_full[cols_to_keep]

    folder_name = f"{int(deltaT)}h" if deltaT >= 1 else f"{round(deltaT * 60)}min"
    os.makedirs(f"{output_path}/{folder_name}", exist_ok = True)

    df_model.to_pickle(f"{output_path}/{folder_name}/mds_icu_preprocessed2_ecg.pkl")
    df_model.to_parquet(f"{output_path}/{folder_name}/mds_icu_preprocessed2_ecg.parquet")

    print(f'Saving result data for {deltaT}')
    df = connection.execute(f"""
                            SELECT {deltaT} AS deltaT_hours, COUNT(*) AS matched_rows, AVG((timedifference_hours * 60) + timedifference_minutes) AS avg_timediff, 
                            MIN((timedifference_hours * 60) + timedifference_minutes) AS min_timediff, MAX((timedifference_hours * 60) + timedifference_minutes) AS max_timediff,
                            ROUND(100.0 * COUNT(*) FILTER (WHERE strat_fold = 0) / COUNT(*), 2) AS train_percentage,
                            ROUND(100.0 * COUNT(*) FILTER (WHERE strat_fold = 1) / COUNT(*), 2) AS validate_percentage,
                            ROUND(100.0 * COUNT(*) FILTER (WHERE strat_fold = 2) / COUNT(*), 2) AS test_percentage,
                            COUNT(bmi_missing) FILTER (bmi_missing = 1) AS bmi_missing_count,
                            COUNT(weight_missing) FILTER (weight_missing = 1) AS weight_missing,
                            COUNT(height_missing) FILTER (height_missing = 1) AS height_missing,
                            MIN(data) AS min_data,
                            MAX(data) AS max_data
                            FROM final
                            """).df()
    results.append(df)
    
    print(f'Saving result data for {deltaT} completed')

    print('Saving matched table')
    if deltaT >= 1: 
        df_model.to_csv(f'{output_path}/{folder_name}/glucose_ecg_pairs_{int(deltaT)}h.csv')
    else: 
        df_model.to_csv(f'{output_path}/{folder_name}/glucose_ecg_pairs_{int(deltaT * 60)}min.csv')
    print('Saving matched table completed')
    
result_table = pd.concat(results, ignore_index=True)
result_table = result_table.sort_values("deltaT_hours", ascending=False)
print(result_table)