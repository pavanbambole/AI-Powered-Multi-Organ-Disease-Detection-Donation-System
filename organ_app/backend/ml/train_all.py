"""
MultiOrganAI - Master Automated Training Pipeline
Executes automated model selection across all three clinical datasets:
- Kidney (UCI Chronic Kidney Disease)
- Heart (UCI Cleveland Heart Disease)
- Liver (Indian Liver Patient Dataset)
Prints Step 12 Final Performance Report.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from automl_pipeline import AutoMLPipeline

DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'data'))

# Kidney target mapping
def map_kidney_target(val):
    clean = str(val).strip().lower()
    if 'notckd' in clean:
        return 0
    elif 'ckd' in clean:
        return 1
    return None

# Liver target mapping (ILPD: 1=disease, 2=healthy)
def map_liver_target(val):
    try:
        v = float(str(val).strip())
        return 1 if v == 1.0 else (0 if v == 2.0 else None)
    except (ValueError, TypeError):
        return None

DATASET_CONFIGS = [
    {
        'organ': 'kidney',
        'csv': os.path.join(DATA_DIR, 'kidney_disease.csv'),
        'features': ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'wbcc'],
        'target_col': 'class',
        'target_map': map_kidney_target
    },
    {
        'organ': 'liver',
        'csv': os.path.join(DATA_DIR, 'liver_disease.csv'),
        'features': [
            'age', 'gender', 'total_bilirubin', 'direct_bilirubin',
            'alkaline_phosphotase', 'alamine_aminotransferase',
            'aspartate_aminotransferase', 'total_proteins', 'albumin',
            'albumin_and_globulin_ratio'
        ],
        'target_col': 'Selector',
        'target_map': map_liver_target
    }
]

def run_pipeline():
    results = {}

    for cfg in DATASET_CONFIGS:
        organ = cfg['organ']
        print(f"\n" + "#" * 70)
        print(f" LAUNCHING AUTOMATED PIPELINE FOR: {organ.upper()}")
        print("#" * 70)

        # For liver dataset, check if column names need renaming from raw ILPD to standard
        csv_file = cfg['csv']
        if organ == 'liver':
            import pandas as pd
            df_l = pd.read_csv(csv_file)
            if 'Age' in df_l.columns or 'TB' in df_l.columns:
                rename_map = {
                    'Age': 'age', 'Gender': 'gender', 'TB': 'total_bilirubin', 'DB': 'direct_bilirubin',
                    'Alkphos': 'alkaline_phosphotase', 'Sgpt': 'alamine_aminotransferase',
                    'Sgot': 'aspartate_aminotransferase', 'TP': 'total_proteins', 'ALB': 'albumin',
                    'A/G Ratio': 'albumin_and_globulin_ratio'
                }
                df_l = df_l.rename(columns=rename_map)
                # Ensure gender is numeric
                if 'gender' in df_l.columns and df_l['gender'].dtype == object:
                    df_l['gender'] = df_l['gender'].astype(str).str.strip().str.lower().map({'male': 1.0, 'female': 0.0}).fillna(1.0)
                temp_csv = os.path.join(DATA_DIR, 'liver_disease_standardized.csv')
                df_l.to_csv(temp_csv, index=False)
                csv_file = temp_csv

        pipeline = AutoMLPipeline(
            organ_name=organ,
            csv_path=csv_file,
            feature_columns=cfg['features'],
            target_column=cfg['target_col'],
            target_mapping=cfg['target_map'],
            random_state=42
        )
        res = pipeline.execute()
        results[organ] = res

    # STEP 12: FINAL TRAINING REPORT
    print("\n" + "=" * 48)
    print("MULTIORGANAI MODEL PERFORMANCE REPORT")
    print("=" * 48)

    for organ in ['kidney', 'liver']:
        r = results[organ]
        print(f"\n{organ.upper()}:")
        print(f"Best Model:    {r['best_model']}")
        print(f"Test Accuracy: {r['test_accuracy']*100:.2f}%")
        print(f"CV Accuracy:   {r['cv_accuracy']*100:.2f}%")
        print(f"F1 Score:      {r['f1_score']*100:.2f}%")

    print("\n" + "=" * 48 + "\n")

    return results

if __name__ == '__main__':
    run_pipeline()
