import os
import sys
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from automl_pipeline import AutoMLPipeline

DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'data'))
RAW_CSV = os.path.join(DATA_DIR, 'liver_disease.csv')

LIVER_FEATURES = [
    'age', 'gender', 'total_bilirubin', 'direct_bilirubin',
    'alkaline_phosphotase', 'alamine_aminotransferase',
    'aspartate_aminotransferase', 'total_proteins', 'albumin',
    'albumin_and_globulin_ratio'
]

def map_liver_target(val):
    try:
        v = float(str(val).strip())
        return 1 if v == 1.0 else (0 if v == 2.0 else None)
    except (ValueError, TypeError):
        return None

def train():
    # Standardize column names if needed
    df_l = pd.read_csv(RAW_CSV)
    if 'Age' in df_l.columns or 'TB' in df_l.columns:
        rename_map = {
            'Age': 'age', 'Gender': 'gender', 'TB': 'total_bilirubin', 'DB': 'direct_bilirubin',
            'Alkphos': 'alkaline_phosphotase', 'Sgpt': 'alamine_aminotransferase',
            'Sgot': 'aspartate_aminotransferase', 'TP': 'total_proteins', 'ALB': 'albumin',
            'A/G Ratio': 'albumin_and_globulin_ratio'
        }
        df_l = df_l.rename(columns=rename_map)
        if 'gender' in df_l.columns and df_l['gender'].dtype == object:
            df_l['gender'] = df_l['gender'].astype(str).str.strip().str.lower().map({'male': 1.0, 'female': 0.0}).fillna(1.0)
        temp_csv = os.path.join(DATA_DIR, 'liver_disease_standardized.csv')
        df_l.to_csv(temp_csv, index=False)
        csv_path = temp_csv
    else:
        csv_path = RAW_CSV

    pipeline = AutoMLPipeline(
        organ_name='liver',
        csv_path=csv_path,
        feature_columns=LIVER_FEATURES,
        target_column='Selector',
        target_mapping=map_liver_target,
        random_state=42
    )
    return pipeline.execute()

if __name__ == '__main__':
    train()
