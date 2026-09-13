import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from automl_pipeline import AutoMLPipeline

DATA_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'data', 'kidney_disease.csv'))

KIDNEY_FEATURES = [
    'age', 'bp', 'sg', 'al', 'su', 'bgr',
    'bu', 'sc', 'sod', 'pot', 'hemo', 'wbcc'
]

def map_kidney_target(val):
    clean = str(val).strip().lower()
    if 'notckd' in clean:
        return 0
    elif 'ckd' in clean:
        return 1
    return None

def train():
    pipeline = AutoMLPipeline(
        organ_name='kidney',
        csv_path=DATA_PATH,
        feature_columns=KIDNEY_FEATURES,
        target_column='class',
        target_mapping=map_kidney_target,
        random_state=42
    )
    return pipeline.execute()

if __name__ == '__main__':
    train()
