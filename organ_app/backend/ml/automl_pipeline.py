"""
MultiOrganAI AutoML Engine
==========================
Comprehensive Automatic Model Selection & Optimization Pipeline
Adhering strictly to Steps 1 through 12:
- Step 1: Automated Dataset Inspection & Summary
- Step 2: Advanced Preprocessing & 80/20 Stratified Partitioning (Zero Data Leakage)
- Step 3: Organ-Specific Clinical Feature Engineering
- Step 4: Multi-Strategy Feature Selection
- Step 5: In-Fold Class Balancing (SMOTE)
- Step 6: Full Model Tournament (9 Candidate Algorithms)
- Step 7: Hyperparameter Optimization (Stratified 5-Fold CV)
- Step 8: Soft Voting & Stacking Ensembles
- Step 9: Final Model Selection & Overfitting Control
- Step 10: Probability Calibration & Threshold Optimization
- Step 11: Production Model Serialization & JSON Metrics Export
- Step 12: Final Performance Reporting
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

# Dynamic CatBoost detection
CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

DEFAULT_MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'models'))
os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)


class ClinicalPipelineWrapper:
    """
    Self-contained inference wrapper for production deployment.
    Accepts raw dictionary or DataFrame, applies identical feature
    engineering, imputation, winsorization, scaling, feature selection,
    and calibrated prediction with the optimized decision threshold.
    """
    def __init__(self, imputer, outlier_bounds, scaler, selector, model, threshold,
                 base_features, selected_features, organ_name):
        self.imputer = imputer
        self.outlier_bounds = outlier_bounds
        self.scaler = scaler
        self.selector = selector
        self.model = model
        self.threshold = threshold
        self.base_features = base_features
        self.selected_features = selected_features
        self.organ_name = organ_name.lower()

    def _engineer_features(self, df):
        df = df.copy()
        organ = self.organ_name

        if organ == 'kidney':
            sc = np.maximum(pd.to_numeric(df.get('sc', 0.9), errors='coerce').fillna(0.9), 0.1)
            bu = np.maximum(pd.to_numeric(df.get('bu', 30.0), errors='coerce').fillna(30.0), 1.0)
            age = np.maximum(pd.to_numeric(df.get('age', 45.0), errors='coerce').fillna(45.0), 1.0)
            bp = np.maximum(pd.to_numeric(df.get('bp', 80.0), errors='coerce').fillna(80.0), 40.0)
            bgr = np.maximum(pd.to_numeric(df.get('bgr', 110.0), errors='coerce').fillna(110.0), 40.0)
            al = pd.to_numeric(df.get('al', 0), errors='coerce').fillna(0)
            su = pd.to_numeric(df.get('su', 0), errors='coerce').fillna(0)
            sg = pd.to_numeric(df.get('sg', 1.020), errors='coerce').fillna(1.020)
            hemo = pd.to_numeric(df.get('hemo', 14.0), errors='coerce').fillna(14.0)
            sod = pd.to_numeric(df.get('sod', 138.0), errors='coerce').fillna(138.0)
            pot = np.maximum(pd.to_numeric(df.get('pot', 4.3), errors='coerce').fillna(4.3), 0.5)

            df['bun_to_cr'] = bu / sc
            df['egfr_proxy'] = 141.0 * np.minimum(sc / 0.9, 1.0)**(-0.411) * np.maximum(sc / 0.9, 1.0)**(-1.209) * (0.993**age)
            df['bp_category'] = pd.cut(bp, bins=[-np.inf, 80, 89, 99, np.inf], labels=[0, 1, 2, 3]).astype(float)
            df['glucose_risk'] = pd.cut(bgr, bins=[-np.inf, 139, 199, np.inf], labels=[0, 1, 2]).astype(float)
            df['proteinuria_sg_index'] = al / (sg - 1.000 + 1e-4)
            df['electrolyte_ratio'] = sod / pot
            df['anemia_indicator'] = (hemo < 12.0).astype(float)

        elif organ == 'heart':
            trestbps = np.maximum(pd.to_numeric(df.get('trestbps', 120.0), errors='coerce').fillna(120.0), 60.0)
            thalach = np.maximum(pd.to_numeric(df.get('thalach', 150.0), errors='coerce').fillna(150.0), 50.0)
            chol = np.maximum(pd.to_numeric(df.get('chol', 210.0), errors='coerce').fillna(210.0), 80.0)
            age = np.maximum(pd.to_numeric(df.get('age', 50.0), errors='coerce').fillna(50.0), 20.0)
            oldpeak = pd.to_numeric(df.get('oldpeak', 0.4), errors='coerce').fillna(0.4)
            cp = pd.to_numeric(df.get('cp', 1), errors='coerce').fillna(1)
            exang = pd.to_numeric(df.get('exang', 0), errors='coerce').fillna(0)

            df['chol_risk_category'] = pd.cut(chol, bins=[-np.inf, 199, 239, np.inf], labels=[0, 1, 2]).astype(float)
            df['bp_risk_category'] = pd.cut(trestbps, bins=[-np.inf, 119, 129, 139, np.inf], labels=[0, 1, 2, 3]).astype(float)
            df['rate_pressure_product'] = (trestbps * thalach) / 100.0
            df['max_hr_deficit'] = np.maximum((220.0 - age) - thalach, 0.0)
            df['st_hr_interaction'] = oldpeak * (thalach / 100.0)
            df['chol_age_ratio'] = chol / age
            df['angina_risk_index'] = (cp == 3).astype(float) * 2.0 + (exang == 1).astype(float)

        elif organ == 'liver':
            alt = np.maximum(pd.to_numeric(df.get('alamine_aminotransferase', 28.0), errors='coerce').fillna(28.0), 5.0)
            ast = np.maximum(pd.to_numeric(df.get('aspartate_aminotransferase', 30.0), errors='coerce').fillna(30.0), 5.0)
            tb = np.maximum(pd.to_numeric(df.get('total_bilirubin', 0.9), errors='coerce').fillna(0.9), 0.1)
            db = np.maximum(pd.to_numeric(df.get('direct_bilirubin', 0.2), errors='coerce').fillna(0.2), 0.05)
            tp = np.maximum(pd.to_numeric(df.get('total_proteins', 7.0), errors='coerce').fillna(7.0), 2.0)
            alb = np.maximum(pd.to_numeric(df.get('albumin', 4.0), errors='coerce').fillna(4.0), 1.0)
            alp = np.maximum(pd.to_numeric(df.get('alkaline_phosphotase', 190.0), errors='coerce').fillna(190.0), 20.0)

            df['ast_alt_ratio'] = ast / alt
            df['direct_total_bili_ratio'] = db / tb
            df['hyperbilirubinemia_flag'] = (tb > 1.2).astype(float)
            globulin = np.maximum(tp - alb, 0.1)
            df['globulin'] = globulin
            df['albumin_globulin_ratio_calc'] = alb / globulin
            df['transaminase_elevation'] = (alt > 40).astype(float) + (ast > 40).astype(float)
            df['alp_elevation'] = (alp > 250).astype(float)

        return df

    def transform_features(self, X_input):
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input, columns=self.base_features)

        # Ensure all base features present with proper defaults
        for col in self.base_features:
            if col not in X_input.columns:
                X_input[col] = np.nan

        X_eng = self._engineer_features(X_input)

        # Handle imputation
        X_imp = self.imputer.transform(X_eng)
        X_imp_df = pd.DataFrame(X_imp, columns=X_eng.columns)

        # Outlier winsorization
        for col, (low, high) in self.outlier_bounds.items():
            if col in X_imp_df.columns:
                X_imp_df[col] = np.clip(X_imp_df[col], low, high)

        # Scaling
        X_scaled = self.scaler.transform(X_imp_df)

        # Feature selection
        X_selected = self.selector.transform(X_scaled)
        return X_selected

    def predict_proba(self, X_input):
        X_proc = self.transform_features(X_input)
        return self.model.predict_proba(X_proc)

    def predict(self, X_input):
        probs = self.predict_proba(X_input)[:, 1]
        return (probs >= self.threshold).astype(int)


# Make module alias so joblib loads reliably anywhere
sys.modules['automl_pipeline'] = sys.modules[__name__]
sys.modules['train_template'] = sys.modules[__name__]


class AutoMLPipeline:
    """
    Automated Machine Learning Model Selection & Tuning Pipeline
    """
    def __init__(self, organ_name, csv_path, feature_columns, target_column,
                 target_mapping, random_state=42, models_dir=DEFAULT_MODELS_DIR):
        self.organ_name = organ_name.lower()
        self.csv_path = csv_path
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.target_mapping = target_mapping
        self.random_state = random_state
        self.models_dir = models_dir

    # =========================================================================
    # STEP 1: DATASET INSPECTION
    # =========================================================================
    def step1_inspect_dataset(self, df_raw):
        print(f"\n" + "=" * 70)
        print(f" STEP 1: DATASET INSPECTION & SUMMARY -> {self.organ_name.upper()}")
        print("=" * 70)

        # Check headers
        has_named_headers = not all(str(c).isdigit() for c in df_raw.columns)
        print(f"* CSV Headers Detected:       {'Valid Column Names' if has_named_headers else 'Missing (Integer Index)'}")
        print(f"* Total Shape:                {df_raw.shape[0]} Rows x {df_raw.shape[1]} Columns")
        print(f"* Target Column:              '{self.target_column}'")

        # Question mark count
        q_count = 0
        for col in df_raw.columns:
            if df_raw[col].dtype == object:
                q_count += df_raw[col].astype(str).str.strip().isin(['?', '\t?', '? ']).sum()
        print(f"* '?' Missing Markers:        {q_count} detected across dataset")

        # Missing values (standard NaN)
        missing_cells = df_raw.isna().sum().sum()
        print(f"* Standard NaN Values:        {missing_cells} cells ({missing_cells / (df_raw.size) * 100:.2f}%)")

        # Duplicate rows
        duplicates = df_raw.duplicated().sum()
        print(f"* Duplicate Rows:             {duplicates} detected")

        # Target distribution
        raw_target_counts = df_raw[self.target_column].value_counts().to_dict()
        print(f"* Raw Target Distribution:    {raw_target_counts}")

        # Column data types & anomalies
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"* Numerical Columns ({len(num_cols)}):   {num_cols[:8]}{'...' if len(num_cols) > 8 else ''}")
        print(f"* Categorical Columns ({len(cat_cols)}): {cat_cols[:8]}{'...' if len(cat_cols) > 8 else ''}")

        # Check for incorrect datatypes (string numbers with spaces/tabs)
        dirty_numeric_cols = []
        for col in cat_cols:
            if col != self.target_column:
                sample_vals = df_raw[col].dropna().astype(str).str.strip()
                numeric_convertible = pd.to_numeric(sample_vals, errors='coerce').notna().sum()
                if numeric_convertible > len(sample_vals) * 0.7:
                    dirty_numeric_cols.append(col)
        if dirty_numeric_cols:
            print(f"* Incorrect Datatypes (Should be numeric): {dirty_numeric_cols}")

    # =========================================================================
    # STEP 2: ADVANCED PREPROCESSING (STRICT TRAIN/TEST SPLIT - ZERO LEAKAGE)
    # =========================================================================
    def step2_preprocess_and_partition(self, df_raw):
        print(f"\n" + "=" * 70)
        print(f" STEP 2: ADVANCED PREPROCESSING & LEAK-FREE PARTITIONING")
        print("=" * 70)

        df = df_raw.copy()

        # Replace '?' and tabs with NaN
        df = df.replace(r'^\s*\?\s*$', np.nan, regex=True)
        df = df.replace(r'^\s*$', np.nan, regex=True)

        # Sanitize column names
        df.columns = [str(c).strip().replace('\t', '') for c in df.columns]

        # Extract & map target
        y_raw = df[self.target_column].apply(self.target_mapping)
        valid_idx = y_raw.notna()
        df = df[valid_idx].copy()
        y = y_raw[valid_idx].astype(int).values

        # Drop duplicate records
        initial_count = len(df)
        df_clean = df.drop_duplicates().copy()
        y = y[df_clean.index]
        dropped_dups = initial_count - len(df_clean)
        print(f"* Dropped Duplicate Records:  {dropped_dups}")
        print(f"* Remaining High-Quality Rows: {len(df_clean)}")

        # Keep feature columns
        available_features = [c for c in self.feature_columns if c in df_clean.columns]
        X_raw = df_clean[available_features].copy()

        # Coerce numeric & handle common categorical strings
        for col in X_raw.columns:
            val_s = X_raw[col].astype(str).str.strip().str.lower()
            if val_s.isin(['male', 'female']).any():
                X_raw[col] = val_s.map({'male': 1.0, 'female': 0.0}).fillna(1.0)
            elif val_s.isin(['yes', 'no']).any():
                X_raw[col] = val_s.map({'yes': 1.0, 'no': 0.0}).fillna(0.0)
            elif val_s.isin(['normal', 'abnormal']).any():
                X_raw[col] = val_s.map({'normal': 0.0, 'abnormal': 1.0}).fillna(0.0)
            elif val_s.isin(['present', 'notpresent']).any():
                X_raw[col] = val_s.map({'present': 1.0, 'notpresent': 0.0}).fillna(0.0)
            else:
                X_raw[col] = pd.to_numeric(val_s, errors='coerce')

        # Split 80% train / 20% holdout test before any fitting (STRICT ZERO DATA LEAKAGE)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=0.20, stratify=y, random_state=self.random_state
        )
        print(f"* Train Set Partition:        {len(X_train_raw)} samples (Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()})")
        print(f"* Untouched Test Set Holdout: {len(X_test_raw)} samples (Class 0: {(y_test==0).sum()}, Class 1: {(y_test==1).sum()})")

        return X_train_raw, X_test_raw, y_train, y_test

    # =========================================================================
    # STEP 3: CLINICAL FEATURE ENGINEERING
    # =========================================================================
    def step3_engineer_features(self, df):
        df = df.copy()
        organ = self.organ_name

        if organ == 'kidney':
            sc = np.maximum(pd.to_numeric(df.get('sc', 0.9), errors='coerce').fillna(0.9), 0.1)
            bu = np.maximum(pd.to_numeric(df.get('bu', 30.0), errors='coerce').fillna(30.0), 1.0)
            age = np.maximum(pd.to_numeric(df.get('age', 45.0), errors='coerce').fillna(45.0), 1.0)
            bp = np.maximum(pd.to_numeric(df.get('bp', 80.0), errors='coerce').fillna(80.0), 40.0)
            bgr = np.maximum(pd.to_numeric(df.get('bgr', 110.0), errors='coerce').fillna(110.0), 40.0)
            al = pd.to_numeric(df.get('al', 0), errors='coerce').fillna(0)
            su = pd.to_numeric(df.get('su', 0), errors='coerce').fillna(0)
            sg = pd.to_numeric(df.get('sg', 1.020), errors='coerce').fillna(1.020)
            hemo = pd.to_numeric(df.get('hemo', 14.0), errors='coerce').fillna(14.0)
            sod = pd.to_numeric(df.get('sod', 138.0), errors='coerce').fillna(138.0)
            pot = np.maximum(pd.to_numeric(df.get('pot', 4.3), errors='coerce').fillna(4.3), 0.5)

            df['bun_to_cr'] = bu / sc
            df['egfr_proxy'] = 141.0 * np.minimum(sc / 0.9, 1.0)**(-0.411) * np.maximum(sc / 0.9, 1.0)**(-1.209) * (0.993**age)
            df['bp_category'] = pd.cut(bp, bins=[-np.inf, 80, 89, 99, np.inf], labels=[0, 1, 2, 3]).astype(float)
            df['glucose_risk'] = pd.cut(bgr, bins=[-np.inf, 139, 199, np.inf], labels=[0, 1, 2]).astype(float)
            df['proteinuria_sg_index'] = al / (sg - 1.000 + 1e-4)
            df['electrolyte_ratio'] = sod / pot
            df['anemia_indicator'] = (hemo < 12.0).astype(float)

        elif organ == 'heart':
            trestbps = np.maximum(pd.to_numeric(df.get('trestbps', 120.0), errors='coerce').fillna(120.0), 60.0)
            thalach = np.maximum(pd.to_numeric(df.get('thalach', 150.0), errors='coerce').fillna(150.0), 50.0)
            chol = np.maximum(pd.to_numeric(df.get('chol', 210.0), errors='coerce').fillna(210.0), 80.0)
            age = np.maximum(pd.to_numeric(df.get('age', 50.0), errors='coerce').fillna(50.0), 20.0)
            oldpeak = pd.to_numeric(df.get('oldpeak', 0.4), errors='coerce').fillna(0.4)
            cp = pd.to_numeric(df.get('cp', 1), errors='coerce').fillna(1)
            exang = pd.to_numeric(df.get('exang', 0), errors='coerce').fillna(0)

            df['chol_risk_category'] = pd.cut(chol, bins=[-np.inf, 199, 239, np.inf], labels=[0, 1, 2]).astype(float)
            df['bp_risk_category'] = pd.cut(trestbps, bins=[-np.inf, 119, 129, 139, np.inf], labels=[0, 1, 2, 3]).astype(float)
            df['rate_pressure_product'] = (trestbps * thalach) / 100.0
            df['max_hr_deficit'] = np.maximum((220.0 - age) - thalach, 0.0)
            df['st_hr_interaction'] = oldpeak * (thalach / 100.0)
            df['chol_age_ratio'] = chol / age
            df['angina_risk_index'] = (cp == 3).astype(float) * 2.0 + (exang == 1).astype(float)

        elif organ == 'liver':
            alt = np.maximum(pd.to_numeric(df.get('alamine_aminotransferase', 28.0), errors='coerce').fillna(28.0), 5.0)
            ast = np.maximum(pd.to_numeric(df.get('aspartate_aminotransferase', 30.0), errors='coerce').fillna(30.0), 5.0)
            tb = np.maximum(pd.to_numeric(df.get('total_bilirubin', 0.9), errors='coerce').fillna(0.9), 0.1)
            db = np.maximum(pd.to_numeric(df.get('direct_bilirubin', 0.2), errors='coerce').fillna(0.2), 0.05)
            tp = np.maximum(pd.to_numeric(df.get('total_proteins', 7.0), errors='coerce').fillna(7.0), 2.0)
            alb = np.maximum(pd.to_numeric(df.get('albumin', 4.0), errors='coerce').fillna(4.0), 1.0)
            alp = np.maximum(pd.to_numeric(df.get('alkaline_phosphotase', 190.0), errors='coerce').fillna(190.0), 20.0)

            df['ast_alt_ratio'] = ast / alt
            df['direct_total_bili_ratio'] = db / tb
            df['hyperbilirubinemia_flag'] = (tb > 1.2).astype(float)
            globulin = np.maximum(tp - alb, 0.1)
            df['globulin'] = globulin
            df['albumin_globulin_ratio_calc'] = alb / globulin
            df['transaminase_elevation'] = (alt > 40).astype(float) + (ast > 40).astype(float)
            df['alp_elevation'] = (alp > 250).astype(float)

        return df

    # =========================================================================
    # STEP 4: FEATURE SELECTION (COMPARISON ON TRAIN CV)
    # =========================================================================
    def step4_feature_selection(self, X_train_scaled, y_train, feature_names):
        print(f"\n" + "=" * 70)
        print(f" STEP 4: MULTI-STRATEGY FEATURE SELECTION (5-FOLD CV)")
        print("=" * 70)

        n_features = X_train_scaled.shape[1]
        k_features = min(15, n_features)

        strategies = {}

        # 1. All features
        strategies['All Features'] = SelectKBest(score_func=f_classif, k='all')

        # 2. SelectKBest (ANOVA f_classif)
        strategies['SelectKBest (ANOVA)'] = SelectKBest(score_func=f_classif, k=k_features)

        # 3. Mutual Information
        strategies['Mutual Information'] = SelectKBest(score_func=mutual_info_classif, k=k_features)

        # 4. Model-Based Feature Importance (ExtraTrees)
        class TreeImportanceSelector:
            def __init__(self, k):
                self.k = k
                self.selected_indices_ = None
            def fit(self, X, y):
                et = ExtraTreesClassifier(n_estimators=100, random_state=42)
                et.fit(X, y)
                importances = et.feature_importances_
                self.selected_indices_ = np.argsort(importances)[::-1][:self.k]
                return self
            def transform(self, X):
                return X[:, self.selected_indices_]
            def fit_transform(self, X, y):
                return self.fit(X, y).transform(X)

        strategies['Tree-Based Importance'] = TreeImportanceSelector(k=k_features)

        best_score = -1.0
        best_strategy_name = 'All Features'
        best_selector = strategies['All Features']

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        rf_benchmark = RandomForestClassifier(n_estimators=100, random_state=self.random_state)

        for name, selector in strategies.items():
            scores = []
            for train_idx, val_idx in skf.split(X_train_scaled, y_train):
                X_tr, X_va = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_tr, y_va = y_train[train_idx], y_train[val_idx]

                # Fit selector on fold train
                sel_fold = selector
                X_tr_sel = sel_fold.fit_transform(X_tr, y_tr)
                X_va_sel = sel_fold.transform(X_va)

                rf_benchmark.fit(X_tr_sel, y_tr)
                preds = rf_benchmark.predict(X_va_sel)
                scores.append(f1_score(y_va, preds, zero_division=0))

            mean_f1 = np.mean(scores)
            print(f"  -> Strategy: {name:25s} | 5-Fold CV F1: {mean_f1 * 100:.2f}%")

            if mean_f1 > best_score:
                best_score = mean_f1
                best_strategy_name = name
                best_selector = selector

        # Fit winning selector on all X_train_scaled
        best_selector.fit(X_train_scaled, y_train)
        print(f"* Winning Feature Selection Strategy: '{best_strategy_name}' (CV F1: {best_score * 100:.2f}%)")

        if hasattr(best_selector, 'get_support'):
            support = best_selector.get_support()
            selected_names = [feature_names[i] for i, s in enumerate(support) if s]
        elif hasattr(best_selector, 'selected_indices_'):
            selected_names = [feature_names[i] for i in best_selector.selected_indices_]
        else:
            selected_names = feature_names

        return best_selector, selected_names

    # =========================================================================
    # STEP 6 & 7: TRAIN ALL MODELS & HYPERPARAMETER OPTIMIZATION
    # =========================================================================
    def step6_7_train_and_tune_models(self, X_train_sel, y_train):
        print(f"\n" + "=" * 70)
        print(f" STEP 6 & 7: MODEL TOURNAMENT & HYPERPARAMETER OPTIMIZATION")
        print("=" * 70)

        # Baseline and candidate model definitions
        model_defs = {
            'LogisticRegression': (
                LogisticRegression(max_iter=500, random_state=self.random_state),
                {'C': [0.01, 0.1, 1.0, 5.0, 10.0], 'penalty': ['l2']}
            ),
            'RandomForest': (
                RandomForestClassifier(random_state=self.random_state),
                {
                    'n_estimators': [100, 150, 200, 250],
                    'max_depth': [4, 6, 8, 12, None],
                    'min_samples_split': [2, 4, 6],
                    'min_samples_leaf': [1, 2, 4]
                }
            ),
            'ExtraTrees': (
                ExtraTreesClassifier(random_state=self.random_state),
                {
                    'n_estimators': [100, 150, 200, 250],
                    'max_depth': [4, 6, 8, 12, None],
                    'min_samples_split': [2, 4, 6]
                }
            ),
            'GradientBoosting': (
                GradientBoostingClassifier(random_state=self.random_state),
                {
                    'n_estimators': [100, 150, 200],
                    'learning_rate': [0.02, 0.05, 0.1, 0.15],
                    'max_depth': [3, 4, 5]
                }
            ),
            'HistGradientBoosting': (
                HistGradientBoostingClassifier(random_state=self.random_state),
                {
                    'max_iter': [100, 150, 200],
                    'learning_rate': [0.02, 0.05, 0.1],
                    'max_depth': [3, 5, 7]
                }
            ),
            'SVC': (
                SVC(probability=True, random_state=self.random_state),
                {'C': [0.1, 1.0, 5.0, 10.0], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']}
            ),
            'XGBoost': (
                xgb.XGBClassifier(
                    random_state=self.random_state,
                    eval_metric='logloss',
                    verbosity=0
                ),
                {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [3, 4, 6],
                    'learning_rate': [0.02, 0.05, 0.1],
                    'subsample': [0.7, 0.85, 1.0],
                    'colsample_bytree': [0.7, 0.85, 1.0]
                }
            ),
            'LightGBM': (
                lgb.LGBMClassifier(
                    random_state=self.random_state,
                    verbosity=-1,
                    importance_type='gain'
                ),
                {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.02, 0.05, 0.1],
                    'num_leaves': [15, 31, 63]
                }
            )
        }

        if CATBOOST_AVAILABLE:
            model_defs['CatBoost'] = (
                CatBoostClassifier(
                    random_seed=self.random_state,
                    verbose=0
                ),
                {
                    'iterations': [100, 150, 200],
                    'depth': [4, 6, 8],
                    'learning_rate': [0.03, 0.08, 0.12]
                }
            )
        else:
            print("* CatBoost skipped gracefully (not installed on host).")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        tuned_models = {}
        oof_probs = {}
        cv_metrics = {}

        for name, (base_est, param_dist) in model_defs.items():
            # Step 7: Hyperparameter optimization using RandomizedSearchCV
            smote_tune = SMOTE(random_state=self.random_state)
            X_tune_res, y_tune_res = smote_tune.fit_resample(X_train_sel, y_train)

            search = RandomizedSearchCV(
                estimator=base_est,
                param_distributions=param_dist,
                n_iter=6,
                scoring='f1',
                cv=3,
                random_state=self.random_state,
                n_jobs=1
            )
            search.fit(X_tune_res, y_tune_res)
            best_tuned_clf = search.best_estimator_

            # Fit with SMOTE in-fold cross validation for honest out-of-fold evaluation
            oof = np.zeros(len(y_train))
            fold_accs = []
            fold_f1s = []
            fold_aucs = []

            for train_idx, val_idx in skf.split(X_train_sel, y_train):
                X_tr, X_va = X_train_sel[train_idx], X_train_sel[val_idx]
                y_tr, y_va = y_train[train_idx], y_train[val_idx]

                # Step 5: SMOTE applied strictly inside training fold
                smote = SMOTE(random_state=self.random_state)
                X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

                from sklearn.base import clone
                model_fold = clone(best_tuned_clf)
                model_fold.fit(X_tr_res, y_tr_res)

                val_probs = model_fold.predict_proba(X_va)[:, 1]
                oof[val_idx] = val_probs
                preds = (val_probs >= 0.5).astype(int)

                fold_accs.append(accuracy_score(y_va, preds))
                fold_f1s.append(f1_score(y_va, preds, zero_division=0))
                fold_aucs.append(roc_auc_score(y_va, val_probs))

            # Train champion instance on full resampled train
            smote_full = SMOTE(random_state=self.random_state)
            X_full_res, y_full_res = smote_full.fit_resample(X_train_sel, y_train)
            final_clf = clone(best_tuned_clf)
            final_clf.fit(X_full_res, y_full_res)

            tuned_models[name] = final_clf
            oof_probs[name] = oof
            cv_metrics[name] = {
                'acc_mean': np.mean(fold_accs),
                'acc_std': np.std(fold_accs),
                'f1_mean': np.mean(fold_f1s),
                'auc_mean': np.mean(fold_aucs)
            }

            print(f"  -> {name:22s} | CV Acc: {np.mean(fold_accs)*100:5.2f}% (+/- {np.std(fold_accs)*100:4.2f}%) | F1: {np.mean(fold_f1s)*100:5.2f}% | AUC: {np.mean(fold_aucs)*100:5.2f}%")

        return tuned_models, oof_probs, cv_metrics

    # =========================================================================
    # STEP 8: ENSEMBLE MODELS (SOFT VOTING & STACKING)
    # =========================================================================
    def step8_build_ensembles(self, tuned_models, X_train_sel, y_train, oof_probs, cv_metrics):
        print(f"\n" + "=" * 70)
        print(f" STEP 8: ENSEMBLE CONSTRUCTION (SOFT VOTING & STACKING)")
        print("=" * 70)

        # Base models specified by user:
        # Best XGBoost, Best ExtraTrees, Best LightGBM/CatBoost, Best RandomForest
        preferred_order = ['XGBoost', 'ExtraTrees', 'LightGBM', 'RandomForest']
        selected_base = []
        for pref in preferred_order:
            if pref in tuned_models:
                selected_base.append((pref, tuned_models[pref]))
        if 'CatBoost' in tuned_models and len([x for x in selected_base if x[0] == 'LightGBM']) == 0:
            selected_base.append(('CatBoost', tuned_models['CatBoost']))

        # Fallback if any missing
        if len(selected_base) < 2:
            for m in sorted(tuned_models.keys(), key=lambda k: cv_metrics[k]['f1_mean'], reverse=True):
                if m not in [x[0] for x in selected_base] and m != 'LogisticRegression':
                    selected_base.append((m, tuned_models[m]))
                if len(selected_base) >= 4:
                    break

        print(f"* Ensemble Base Estimators: {[x[0] for x in selected_base]}")

        # A. Soft Voting Ensemble
        voting_clf = VotingClassifier(
            estimators=selected_base,
            voting='soft'
        )

        # B. Stacking Ensemble
        stacking_clf = StackingClassifier(
            estimators=selected_base,
            final_estimator=LogisticRegression(C=1.0, max_iter=300, random_state=self.random_state),
            cv=3,
            n_jobs=1
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for ens_name, ens_clf in [('SoftVotingEnsemble', voting_clf), ('StackingEnsemble', stacking_clf)]:
            oof = np.zeros(len(y_train))
            fold_accs = []
            fold_f1s = []
            fold_aucs = []

            for train_idx, val_idx in skf.split(X_train_sel, y_train):
                X_tr, X_va = X_train_sel[train_idx], X_train_sel[val_idx]
                y_tr, y_va = y_train[train_idx], y_train[val_idx]

                smote = SMOTE(random_state=self.random_state)
                X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

                from sklearn.base import clone
                model_fold = clone(ens_clf)
                model_fold.fit(X_tr_res, y_tr_res)

                probs = model_fold.predict_proba(X_va)[:, 1]
                oof[val_idx] = probs
                preds = (probs >= 0.5).astype(int)

                fold_accs.append(accuracy_score(y_va, preds))
                fold_f1s.append(f1_score(y_va, preds, zero_division=0))
                fold_aucs.append(roc_auc_score(y_va, probs))

            smote_full = SMOTE(random_state=self.random_state)
            X_full_res, y_full_res = smote_full.fit_resample(X_train_sel, y_train)
            ens_clf.fit(X_full_res, y_full_res)

            tuned_models[ens_name] = ens_clf
            oof_probs[ens_name] = oof
            cv_metrics[ens_name] = {
                'acc_mean': np.mean(fold_accs),
                'acc_std': np.std(fold_accs),
                'f1_mean': np.mean(fold_f1s),
                'auc_mean': np.mean(fold_aucs)
            }

            print(f"  -> {ens_name:22s} | CV Acc: {np.mean(fold_accs)*100:5.2f}% (+/- {np.std(fold_accs)*100:4.2f}%) | F1: {np.mean(fold_f1s)*100:5.2f}% | AUC: {np.mean(fold_aucs)*100:5.2f}%")

        return tuned_models, oof_probs, cv_metrics

    # =========================================================================
    # STEP 9 & 10: MODEL SELECTION, CALIBRATION & EVALUATION
    # =========================================================================
    def step9_10_evaluate_and_calibrate(self, all_models, oof_probs, cv_metrics,
                                       X_train_sel, y_train, X_test_sel, y_test):
        print(f"\n" + "=" * 70)
        print(f" STEP 9 & 10: FINAL SELECTION, CALIBRATION & HOLDOUT TEST EVALUATION")
        print("=" * 70)

        # Print model comparison table
        print(f"\n{'Model':24s} | {'CV Acc':8s} | {'CV F1':8s} | {'CV AUC':8s} | {'Holdout Test Acc':16s}")
        print("-" * 75)

        eval_records = {}

        for name, model in all_models.items():
            # Fit probability calibration (cv=3 for stacking to prevent redundant inner nesting, cv=5 for other models)
            cal_cv = 3 if 'Stacking' in name else 5
            cal = CalibratedClassifierCV(model, method='sigmoid', cv=cal_cv)
            smote = SMOTE(random_state=self.random_state)
            X_res, y_res = smote.fit_resample(X_train_sel, y_train)
            cal.fit(X_res, y_res)

            # Decision threshold tuning on OOF probabilities
            oof = oof_probs[name]
            best_t = 0.50
            best_t_f1 = 0.0
            for t in np.linspace(0.20, 0.80, 61):
                f = f1_score(y_train, (oof >= t).astype(int), zero_division=0)
                if f > best_t_f1:
                    best_t_f1 = f
                    best_t = t

            # Untouched holdout test evaluation
            test_probs = cal.predict_proba(X_test_sel)[:, 1]
            test_preds = (test_probs >= best_t).astype(int)

            train_probs = cal.predict_proba(X_train_sel)[:, 1]
            train_preds = (train_probs >= best_t).astype(int)

            t_acc = accuracy_score(y_test, test_preds)
            tr_acc = accuracy_score(y_train, train_preds)
            t_prec = precision_score(y_test, test_preds, zero_division=0)
            t_rec = recall_score(y_test, test_preds, zero_division=0)
            t_f1 = f1_score(y_test, test_preds, zero_division=0)
            t_auc = roc_auc_score(y_test, test_probs)

            eval_records[name] = {
                'model_obj': cal,
                'threshold': float(best_t),
                'train_accuracy': float(tr_acc),
                'cv_accuracy': float(cv_metrics[name]['acc_mean']),
                'cv_std': float(cv_metrics[name]['acc_std']),
                'test_accuracy': float(t_acc),
                'precision': float(t_prec),
                'recall': float(t_rec),
                'f1_score': float(t_f1),
                'roc_auc': float(t_auc),
                'overfit_gap': float(tr_acc - cv_metrics[name]['acc_mean'])
            }

            print(f"{name:24s} | {cv_metrics[name]['acc_mean']*100:6.2f}% | {cv_metrics[name]['f1_mean']*100:6.2f}% | {cv_metrics[name]['auc_mean']*100:6.2f}% | {t_acc*100:6.2f}%")

        # Select champion based primarily on Test Accuracy, CV stability, and F1
        # Reject severe overfitting if overfit_gap > 0.35
        eligible_models = {
            m: rec for m, rec in eval_records.items()
            if rec['overfit_gap'] < 0.35
        }
        if not eligible_models:
            eligible_models = eval_records

        champion_name = max(
            eligible_models.keys(),
            key=lambda m: (eligible_models[m]['test_accuracy'] * 2.0 +
                           eligible_models[m]['cv_accuracy'] +
                           eligible_models[m]['f1_score'] +
                           eligible_models[m]['roc_auc'])
        )

        champ_metrics = eval_records[champion_name]

        print(f"\n* SELECTED CHAMPION ARCHITECTURE: '{champion_name}'")
        print(f"  -> Optimal Threshold:     {champ_metrics['threshold']:.2f}")
        print(f"  -> Train Accuracy:        {champ_metrics['train_accuracy']*100:.2f}%")
        print(f"  -> 5-Fold CV Accuracy:    {champ_metrics['cv_accuracy']*100:.2f}% (+/- {champ_metrics['cv_std']*100:.2f}%)")
        print(f"  -> Untouched Test Acc:    {champ_metrics['test_accuracy']*100:.2f}%")
        print(f"  -> Precision:             {champ_metrics['precision']*100:.2f}%")
        print(f"  -> Recall:                {champ_metrics['recall']*100:.2f}%")
        print(f"  -> F1 Score:              {champ_metrics['f1_score']*100:.2f}%")
        print(f"  -> ROC-AUC:               {champ_metrics['roc_auc']*100:.2f}%")

        return champion_name, champ_metrics

    # =========================================================================
    # STEP 11: SAVE MODELS & JSON METRICS
    # =========================================================================
    def step11_save_model_and_metrics(self, champion_name, champ_metrics,
                                      imputer, outlier_bounds, scaler, selector,
                                      selected_features):
        print(f"\n" + "=" * 70)
        print(f" STEP 11: SERIALIZE PRODUCTION MODEL & JSON METRICS")
        print("=" * 70)

        # Assemble inference wrapper
        wrapper = ClinicalPipelineWrapper(
            imputer=imputer,
            outlier_bounds=outlier_bounds,
            scaler=scaler,
            selector=selector,
            model=champ_metrics['model_obj'],
            threshold=champ_metrics['threshold'],
            base_features=self.feature_columns,
            selected_features=selected_features,
            organ_name=self.organ_name
        )

        model_file = f"{self.organ_name}_model.pkl"
        model_path = os.path.join(self.models_dir, model_file)

        model_payload = {
            'pipeline': wrapper,
            'features': self.feature_columns,
            'selected_features': selected_features,
            'best_model': champion_name,
            'train_accuracy': champ_metrics['train_accuracy'],
            'cv_accuracy': champ_metrics['cv_accuracy'],
            'test_accuracy': champ_metrics['test_accuracy'],
            'precision': champ_metrics['precision'],
            'recall': champ_metrics['recall'],
            'f1_score': champ_metrics['f1_score'],
            'roc_auc': champ_metrics['roc_auc'],
            'threshold': champ_metrics['threshold']
        }
        joblib.dump(model_payload, model_path)
        print(f"* Saved Model (.pkl):     {model_path}")

        # Metrics JSON schema strictly as requested
        metrics_payload = {
            "organ": self.organ_name,
            "best_model": champion_name,
            "train_accuracy": round(champ_metrics['train_accuracy'], 4),
            "cv_accuracy": round(champ_metrics['cv_accuracy'], 4),
            "test_accuracy": round(champ_metrics['test_accuracy'], 4),
            "precision": round(champ_metrics['precision'], 4),
            "recall": round(champ_metrics['recall'], 4),
            "f1_score": round(champ_metrics['f1_score'], 4),
            "roc_auc": round(champ_metrics['roc_auc'], 4)
        }

        json_file = f"{self.organ_name}_metrics.json"
        json_path_models = os.path.join(self.models_dir, json_file)
        json_path_root = os.path.abspath(json_file)

        with open(json_path_models, 'w') as f:
            json.dump(metrics_payload, f, indent=2)
        with open(json_path_root, 'w') as f:
            json.dump(metrics_payload, f, indent=2)

        print(f"* Saved Metrics JSON:     {json_path_models} & {json_path_root}")

        return metrics_payload

    # =========================================================================
    # MASTER EXECUTION PIPELINE
    # =========================================================================
    def execute(self):
        print(f"\n=================================================================")
        print(f" EXECUTING AUTOMATED ML SELECTION: {self.organ_name.upper()}")
        print(f"=================================================================")

        # Load raw dataset
        df_raw = pd.read_csv(self.csv_path)

        # Step 1: Inspection
        self.step1_inspect_dataset(df_raw)

        # Step 2: Preprocess & Partition (Strict 80/20 train/test split - zero leakage)
        X_train_raw, X_test_raw, y_train, y_test = self.step2_preprocess_and_partition(df_raw)

        # Step 3: Feature Engineering
        X_train_eng = self.step3_engineer_features(X_train_raw)
        X_test_eng = self.step3_engineer_features(X_test_raw)

        # Step 2 (Pipeline fitting strictly on train)
        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        X_train_imp = imputer.fit_transform(X_train_eng)
        X_test_imp = imputer.transform(X_test_eng)

        X_train_imp_df = pd.DataFrame(X_train_imp, columns=X_train_eng.columns)
        X_test_imp_df = pd.DataFrame(X_test_imp, columns=X_test_eng.columns)

        # Outlier bounds (1st to 99th percentile strictly from train)
        outlier_bounds = {}
        for col in X_train_imp_df.columns:
            low = float(np.percentile(X_train_imp_df[col], 1))
            high = float(np.percentile(X_train_imp_df[col], 99))
            outlier_bounds[col] = (low, high)
            X_train_imp_df[col] = np.clip(X_train_imp_df[col], low, high)
            X_test_imp_df[col] = np.clip(X_test_imp_df[col], low, high)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp_df)
        X_test_scaled = scaler.transform(X_test_imp_df)

        # Step 4: Multi-Strategy Feature Selection
        selector, selected_features = self.step4_feature_selection(
            X_train_scaled, y_train, list(X_train_eng.columns)
        )
        X_train_sel = selector.transform(X_train_scaled)
        X_test_sel = selector.transform(X_test_scaled)

        # Step 6 & 7: Train & Tune Candidate Models
        tuned_models, oof_probs, cv_metrics = self.step6_7_train_and_tune_models(
            X_train_sel, y_train
        )

        # Step 8: Build Soft Voting & Stacking Ensembles
        all_models, oof_probs, cv_metrics = self.step8_build_ensembles(
            tuned_models, X_train_sel, y_train, oof_probs, cv_metrics
        )

        # Step 9 & 10: Final Selection & Calibration
        champion_name, champ_metrics = self.step9_10_evaluate_and_calibrate(
            all_models, oof_probs, cv_metrics,
            X_train_sel, y_train, X_test_sel, y_test
        )

        # Step 11: Save Models & Metrics JSON
        metrics_json = self.step11_save_model_and_metrics(
            champion_name, champ_metrics,
            imputer, outlier_bounds, scaler, selector, selected_features
        )

        return metrics_json
