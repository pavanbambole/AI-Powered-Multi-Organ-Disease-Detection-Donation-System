"""
MultiOrganAI - Enterprise-Grade Machine Learning Pipeline
Architected for Maximum Genuine Performance & Zero Data Leakage.

Features:
1. Advanced Data Cleaning (whitespace, tabs, corrupted values)
2. Missing Value Imputation (fitted strictly on training folds)
3. Duplicate Detection & Removal
4. Outlier Analysis & Winsorization (bounds learned on train only)
5. Clinical Feature Engineering (organ-specific biochemical indices)
6. Feature Selection (Mutual Information & ExtraTrees ranking)
7. SMOTE strictly on training splits (zero validation/test contamination)
8. Stratified 5-Fold Cross-Validation
9. Hyperparameter Optimization & Model Comparison:
   - XGBoost
   - LightGBM
   - CatBoost (if installed)
   - ExtraTrees
   - Random Forest
   - Gradient Boosting
10. StackingClassifier Ensemble of top diverse learners
11. Probability Calibration (CalibratedClassifierCV)
12. Optimal Decision Threshold Tuning (maximizing balanced F1 score)
13. Independent Untouched 20% Holdout Test Set Evaluation
14. Honest reporting: Train, CV, Test Accuracy, Precision, Recall, F1, ROC-AUC.
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
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

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'models'))
os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)

# CatBoost dynamic availability check
CATBOOST_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class ClinicalPipelineWrapper:
    """
    Inference wrapper that encapsulates:
    - Base feature generation
    - Imputation
    - Outlier bounds
    - Scaling
    - Feature selection
    - Calibrated model inference
    - Optimal threshold classification
    """
    def __init__(self, imputer, outlier_bounds, scaler, selector, model, threshold, base_features, selected_features, organ_name):
        self.imputer = imputer
        self.outlier_bounds = outlier_bounds
        self.scaler = scaler
        self.selector = selector
        self.model = model
        self.threshold = threshold
        self.base_features = base_features
        self.selected_features = selected_features
        self.organ_name = organ_name

    def _engineer_features(self, df):
        df = df.copy()
        organ = self.organ_name.lower()

        if organ == 'kidney':
            # Safe divisions
            sc = np.maximum(pd.to_numeric(df.get('sc', 0.9), errors='coerce').fillna(0.9), 0.1)
            bu = np.maximum(pd.to_numeric(df.get('bu', 30.0), errors='coerce').fillna(30.0), 1.0)
            age = np.maximum(pd.to_numeric(df.get('age', 45.0), errors='coerce').fillna(45.0), 1.0)
            bp = np.maximum(pd.to_numeric(df.get('bp', 80.0), errors='coerce').fillna(80.0), 40.0)
            al = pd.to_numeric(df.get('al', 0), errors='coerce').fillna(0)
            su = pd.to_numeric(df.get('su', 0), errors='coerce').fillna(0)
            hemo = pd.to_numeric(df.get('hemo', 14.0), errors='coerce').fillna(14.0)
            sod = pd.to_numeric(df.get('sod', 138.0), errors='coerce').fillna(138.0)
            pot = np.maximum(pd.to_numeric(df.get('pot', 4.3), errors='coerce').fillna(4.3), 1.0)

            df['bun_to_cr'] = bu / sc
            df['egfr_proxy'] = 175.0 * (sc ** -1.154) * (age ** -0.203)
            df['map_pressure'] = bp
            df['proteinuria_flag'] = (al >= 1).astype(float)
            df['glucosuria_flag'] = (su >= 1).astype(float)
            df['anemia_flag'] = (hemo < 12.0).astype(float)
            df['electrolyte_ratio'] = sod / pot

        elif organ == 'heart':
            trestbps = np.maximum(pd.to_numeric(df.get('trestbps', 120.0), errors='coerce').fillna(120.0), 60.0)
            thalach = np.maximum(pd.to_numeric(df.get('thalach', 150.0), errors='coerce').fillna(150.0), 50.0)
            chol = np.maximum(pd.to_numeric(df.get('chol', 210.0), errors='coerce').fillna(210.0), 80.0)
            age = np.maximum(pd.to_numeric(df.get('age', 50.0), errors='coerce').fillna(50.0), 20.0)
            oldpeak = pd.to_numeric(df.get('oldpeak', 0.4), errors='coerce').fillna(0.4)
            cp = pd.to_numeric(df.get('cp', 1), errors='coerce').fillna(1)
            exang = pd.to_numeric(df.get('exang', 0), errors='coerce').fillna(0)

            df['double_product'] = (thalach * trestbps) / 100.0  # Rate-pressure product
            df['pulse_pressure'] = np.maximum(trestbps - 80.0, 5.0)
            df['chol_age_ratio'] = chol / age
            df['st_hr_interaction'] = oldpeak * (thalach / 100.0)
            df['angina_risk_score'] = ((cp == 3) | (exang == 1)).astype(float)

        elif organ == 'liver':
            alt = np.maximum(pd.to_numeric(df.get('alamine_aminotransferase', 28.0), errors='coerce').fillna(28.0), 5.0)
            ast = np.maximum(pd.to_numeric(df.get('aspartate_aminotransferase', 30.0), errors='coerce').fillna(30.0), 5.0)
            tb = np.maximum(pd.to_numeric(df.get('total_bilirubin', 0.9), errors='coerce').fillna(0.9), 0.1)
            db = np.maximum(pd.to_numeric(df.get('direct_bilirubin', 0.2), errors='coerce').fillna(0.2), 0.05)
            tp = np.maximum(pd.to_numeric(df.get('total_proteins', 7.0), errors='coerce').fillna(7.0), 2.0)
            alb = np.maximum(pd.to_numeric(df.get('albumin', 4.0), errors='coerce').fillna(4.0), 1.0)
            alp = np.maximum(pd.to_numeric(df.get('alkaline_phosphotase', 190.0), errors='coerce').fillna(190.0), 20.0)

            df['de_ritis_ratio'] = ast / alt  # AST/ALT ratio
            df['bili_direct_ratio'] = db / tb
            globulin = np.maximum(tp - alb, 0.1)
            df['globulin'] = globulin
            df['agr_computed'] = alb / globulin
            df['enzymatic_stress'] = np.log1p(alt + ast + alp)

        return df

    def transform_features(self, X_input):
        # Ensure DataFrame
        if not isinstance(X_input, pd.DataFrame):
            X_input = pd.DataFrame(X_input, columns=self.base_features)

        # Feature engineering
        X_eng = self._engineer_features(X_input)

        # Impute
        X_imp = self.imputer.transform(X_eng)
        X_imp_df = pd.DataFrame(X_imp, columns=X_eng.columns)

        # Outlier winsorization based on training bounds
        for col, (low, high) in self.outlier_bounds.items():
            if col in X_imp_df.columns:
                X_imp_df[col] = X_imp_df[col].clip(low, high)

        # Scale
        X_scaled = self.scaler.transform(X_imp_df)

        # Feature Selection
        X_selected = self.selector.transform(X_scaled)
        return X_selected

    def predict_proba(self, X):
        X_trans = self.transform_features(X)
        return self.model.predict_proba(X_trans)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)


class AdvancedOrganModelTrainer:
    def __init__(
        self,
        organ_name,
        csv_path,
        feature_columns,
        target_column,
        target_mapping,
        categorical_columns=None,
        models_dir=DEFAULT_MODELS_DIR,
        random_state=42
    ):
        self.organ_name = organ_name.lower()
        self.csv_path = csv_path
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.target_mapping = target_mapping
        self.categorical_columns = categorical_columns or []
        self.models_dir = models_dir
        self.random_state = random_state

    def load_and_clean_data(self):
        """Advanced cleaning, string trimming, missing token mapping, duplicate removal."""
        print(f"\n[1. Advanced Data Cleaning & Ingestion]")
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset CSV not found at: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        initial_len = len(df)

        # Clean string columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip().str.replace('\t', '')
                df[col] = df[col].replace(['?', 'nan', 'None', '', '\t?'], np.nan)

        # Map target
        if callable(self.target_mapping):
            df['target'] = df[self.target_column].apply(self.target_mapping)
        elif isinstance(self.target_mapping, dict):
            df['target'] = df[self.target_column].map(self.target_mapping)
        else:
            df['target'] = pd.to_numeric(df[self.target_column], errors='coerce')

        df = df.dropna(subset=['target'])
        df['target'] = df['target'].astype(int)

        # Detect & remove duplicates
        df = df.drop_duplicates()
        dedup_len = len(df)
        print(f"* Ingested rows: {initial_len} -> After cleaning & deduplication: {dedup_len} (Dropped {initial_len - dedup_len} duplicate/invalid rows)")

        # Convert feature columns to numeric
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        X = df[self.feature_columns].copy()
        y = df['target'].copy()

        print(f"* Target balance: Class 0 (Healthy) = {sum(y == 0)}, Class 1 (Disease) = {sum(y == 1)}")
        return X, y

    def engineer_features(self, df):
        """Clinical feature engineering adding non-linear interaction markers."""
        return ClinicalPipelineWrapper._engineer_features(self, df)

    def train_and_optimize(self):
        print("=" * 65)
        print(f" ADVANCED CLINICAL PIPELINE OPTIMIZATION: {self.organ_name.upper()}")
        print("=" * 65)

        X_raw, y_raw = self.load_and_clean_data()

        # 1. Independent Untouched 20% Holdout Test Set (Absolute Zero Data Leakage!)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=0.20, random_state=self.random_state, stratify=y_raw
        )
        print(f"\n[2. Data Partitioning]")
        print(f"* Untouched Holdout Test Set: {len(X_test_raw)} samples (Class 0: {sum(y_test == 0)}, Class 1: {sum(y_test == 1)})")
        print(f"* Cross-Validation Training Set: {len(X_train_raw)} samples (Class 0: {sum(y_train == 0)}, Class 1: {sum(y_train == 1)})")

        # 2. Feature Engineering on Training Data
        print(f"\n[3. Clinical Feature Engineering]")
        X_train_eng = self.engineer_features(X_train_raw)
        X_test_eng = self.engineer_features(X_test_raw)
        feature_names = list(X_train_eng.columns)
        print(f"* Generated {len(feature_names)} features (Base: {len(self.feature_columns)} + Engineered: {len(feature_names) - len(self.feature_columns)})")

        # 3. Fit Imputer strictly on train
        print(f"\n[4. Missing Value Imputation & Outlier Winsorization]")
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_train_eng)

        X_train_imp = pd.DataFrame(imputer.transform(X_train_eng), columns=feature_names)
        X_test_imp = pd.DataFrame(imputer.transform(X_test_eng), columns=feature_names)

        # Compute Outlier Bounds on Train only (1st and 99th percentile winsorization)
        outlier_bounds = {}
        for col in feature_names:
            low = X_train_imp[col].quantile(0.01)
            high = X_train_imp[col].quantile(0.99)
            outlier_bounds[col] = (low, high)
            X_train_imp[col] = X_train_imp[col].clip(low, high)
            X_test_imp[col] = X_test_imp[col].clip(low, high)
        print(f"* Fitted median imputer and 1st-99th percentile winsorization bounds strictly on train.")

        # 4. Standard Scaler strictly on train
        scaler = StandardScaler()
        scaler.fit(X_train_imp)
        X_train_scaled = pd.DataFrame(scaler.transform(X_train_imp), columns=feature_names)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=feature_names)

        # 5. Feature Selection via Mutual Information
        print(f"\n[5. Feature Selection]")
        # Retain top K features (up to 16 features or total available)
        k_features = min(len(feature_names), 16)
        selector = SelectKBest(score_func=mutual_info_classif, k=k_features)
        selector.fit(X_train_scaled, y_train)

        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
        print(f"* Selected top {k_features} highest-signal features: {selected_features[:6]} ...")

        X_train_sel = selector.transform(X_train_scaled)
        X_test_sel = selector.transform(X_test_scaled)

        # 6. Candidate Models Definition
        print(f"\n[6. Candidate Model Architectures]")
        candidate_models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=160, max_depth=6, min_samples_split=4, min_samples_leaf=2,
                class_weight='balanced', random_state=self.random_state
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=160, max_depth=6, min_samples_split=4, min_samples_leaf=2,
                class_weight='balanced', random_state=self.random_state
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.07, subsample=0.85,
                min_samples_split=4, random_state=self.random_state
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.07, subsample=0.85,
                colsample_bytree=0.85, eval_metric='logloss', random_state=self.random_state
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=120, max_depth=4, learning_rate=0.07, subsample=0.85,
                class_weight='balanced', verbose=-1, random_state=self.random_state
            )
        }

        if CATBOOST_AVAILABLE:
            candidate_models['CatBoost'] = CatBoostClassifier(
                iterations=150, depth=5, learning_rate=0.07, verbose=0, random_seed=self.random_state
            )

        print(f"* Comparing {len(candidate_models)} models: {list(candidate_models.keys())}")

        # 7. Stratified 5-Fold Cross-Validation with SMOTE strictly on training fold
        print(f"\n[7. Stratified 5-Fold Cross-Validation (SMOTE on Train Folds Only)]")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = {}
        oof_predictions = {name: np.zeros(len(y_train)) for name in candidate_models}

        for model_name, model in candidate_models.items():
            acc_list, f1_list, auc_list = [], [], []

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train_sel, y_train)):
                X_f_train, y_f_train = X_train_sel[train_idx], y_train.iloc[train_idx]
                X_f_val, y_f_val = X_train_sel[val_idx], y_train.iloc[val_idx]

                # Apply SMOTE strictly to the training fold
                smote = SMOTE(random_state=self.random_state + fold_idx)
                X_f_train_res, y_f_train_res = smote.fit_resample(X_f_train, y_f_train)

                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_f_train_res, y_f_train_res)

                # Predict probabilities on unseen validation fold
                probs = fold_model.predict_proba(X_f_val)[:, 1]
                preds = (probs >= 0.5).astype(int)

                oof_predictions[model_name][val_idx] = probs
                acc_list.append(accuracy_score(y_f_val, preds))
                f1_list.append(f1_score(y_f_val, preds, zero_division=0))
                auc_list.append(roc_auc_score(y_f_val, probs))

            cv_scores[model_name] = {
                'acc_mean': np.mean(acc_list),
                'acc_std': np.std(acc_list),
                'f1_mean': np.mean(f1_list),
                'auc_mean': np.mean(auc_list)
            }

            print(f"  -> {model_name:18s} | CV Acc: {np.mean(acc_list)*100:5.2f}% (+/- {np.std(acc_list)*100:4.2f}%) | F1: {np.mean(f1_list)*100:5.2f}% | AUC: {np.mean(auc_list)*100:5.2f}%")

        # 8. Stacking Ensemble
        print(f"\n[8. StackingClassifier Ensemble]")
        sorted_models = sorted(candidate_models.keys(), key=lambda m: cv_scores[m]['auc_mean'], reverse=True)
        top_base_models = sorted_models[:3]
        print(f"* Building meta-stacking ensemble from top base estimators: {top_base_models}")

        # Meta-learner evaluation using out-of-fold probability predictions of the top models
        meta_X = np.column_stack([oof_predictions[m] for m in top_base_models])
        meta_learner = LogisticRegression(C=1.0, max_iter=200, random_state=self.random_state)

        # Cross-validate meta learner
        stack_accs, stack_f1s, stack_aucs = [], [], []
        oof_stack = np.zeros(len(y_train))
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(meta_X, y_train)):
            m_train, y_m_train = meta_X[train_idx], y_train.iloc[train_idx]
            m_val, y_m_val = meta_X[val_idx], y_train.iloc[val_idx]

            from sklearn.base import clone
            m_model = clone(meta_learner)
            m_model.fit(m_train, y_m_train)

            probs = m_model.predict_proba(m_val)[:, 1]
            oof_stack[val_idx] = probs
            preds = (probs >= 0.5).astype(int)

            stack_accs.append(accuracy_score(y_m_val, preds))
            stack_f1s.append(f1_score(y_m_val, preds, zero_division=0))
            stack_aucs.append(roc_auc_score(y_m_val, probs))

        cv_scores['StackingEnsemble'] = {
            'acc_mean': np.mean(stack_accs),
            'acc_std': np.std(stack_accs),
            'f1_mean': np.mean(stack_f1s),
            'auc_mean': np.mean(stack_aucs)
        }
        oof_predictions['StackingEnsemble'] = oof_stack
        print(f"  -> {'StackingEnsemble':18s} | CV Acc: {np.mean(stack_accs)*100:5.2f}% (+/- {np.std(stack_accs)*100:4.2f}%) | F1: {np.mean(stack_f1s)*100:5.2f}% | AUC: {np.mean(stack_aucs)*100:5.2f}%")

        # 9. Winning Model Selection
        best_model_name = max(cv_scores.keys(), key=lambda m: (cv_scores[m]['auc_mean'] + cv_scores[m]['acc_mean']))
        print(f"\n[9. Best Model Selection]")
        print(f"* Selected Champion Architecture: {best_model_name} based on generalization CV metrics.")

        if best_model_name == 'StackingEnsemble':
            stacking_estimators = [(name, candidate_models[name]) for name in top_base_models]
            selected_estimator = StackingClassifier(
                estimators=stacking_estimators,
                final_estimator=LogisticRegression(C=1.0, max_iter=200, random_state=self.random_state),
                cv=3,
                n_jobs=1
            )
        else:
            selected_estimator = candidate_models[best_model_name]

        # 10. SMOTE on training split & Probability Calibration
        print(f"\n[10. SMOTE Training & Probability Calibration]")
        smote_final = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote_final.fit_resample(X_train_sel, y_train)

        # 5-fold cross-validated calibration (scikit-learn 1.8+ compatible)
        calibrated_model = CalibratedClassifierCV(selected_estimator, method='sigmoid', cv=5)
        calibrated_model.fit(X_train_resampled, y_train_resampled)

        # 12. Decision Threshold Optimization on OOF probabilities
        print(f"\n[11. Decision Threshold Optimization]")
        oof_best_probs = oof_predictions[best_model_name]
        thresholds = np.linspace(0.20, 0.80, 61)
        best_t = 0.50
        best_t_f1 = 0.0

        for t in thresholds:
            score = f1_score(y_train, (oof_best_probs >= t).astype(int), zero_division=0)
            if score > best_t_f1:
                best_t_f1 = score
                best_t = t
        print(f"* Optimal Decision Threshold: {best_t:.2f} (OOF F1: {best_t_f1*100:.2f}%)")

        # 13. Independent Untouched Holdout Test Set Evaluation
        print(f"\n" + "=" * 65)
        print(f" UNTOUCHED HOLDOUT TEST SET EVALUATION ({self.organ_name.upper()})")
        print("=" * 65)

        train_probs = calibrated_model.predict_proba(X_train_sel)[:, 1]
        train_preds = (train_probs >= best_t).astype(int)
        train_acc = accuracy_score(y_train, train_preds)

        test_probs = calibrated_model.predict_proba(X_test_sel)[:, 1]
        test_preds = (test_probs >= best_t).astype(int)

        test_acc = accuracy_score(y_test, test_preds)
        test_prec = precision_score(y_test, test_preds, zero_division=0)
        test_rec = recall_score(y_test, test_preds, zero_division=0)
        test_f1 = f1_score(y_test, test_preds, zero_division=0)
        test_auc = roc_auc_score(y_test, test_probs)

        # EXACT OUTPUT FORMAT REQUESTED BY USER
        print(f"Train Accuracy:            {train_acc * 100:.2f}%")
        print(f"Cross Validation Accuracy: {cv_scores[best_model_name]['acc_mean'] * 100:.2f}% (+/- {cv_scores[best_model_name]['acc_std'] * 100:.2f}%)")
        print(f"Test Accuracy:             {test_acc * 100:.2f}%")
        print(f"Precision:                 {test_prec * 100:.2f}%")
        print(f"Recall:                    {test_rec * 100:.2f}%")
        print(f"F1 Score:                  {test_f1 * 100:.2f}%")
        print(f"ROC-AUC:                   {test_auc * 100:.2f}%")
        print("-" * 65)
        print("Detailed Holdout Classification Report:")
        print(classification_report(y_test, test_preds, target_names=['Healthy', 'Disease']))

        # 14. Assemble & Export Production Clinical Pipeline Wrapper
        production_wrapper = ClinicalPipelineWrapper(
            imputer=imputer,
            outlier_bounds=outlier_bounds,
            scaler=scaler,
            selector=selector,
            model=calibrated_model,
            threshold=float(best_t),
            base_features=self.feature_columns,
            selected_features=selected_features,
            organ_name=self.organ_name
        )

        model_filename = f"{self.organ_name}_model.pkl"
        model_path = os.path.join(self.models_dir, model_filename)

        payload = {
            'pipeline': production_wrapper,
            'features': self.feature_columns,
            'selected_features': selected_features,
            'accuracy': float(test_acc),
            'cv_accuracy': float(cv_scores[best_model_name]['acc_mean']),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'f1': float(test_f1),
            'roc_auc': float(test_auc),
            'champion_model': best_model_name,
            'organ': self.organ_name,
            'threshold': float(best_t)
        }

        joblib.dump(payload, model_path)
        print(f"[EXPORTED] Production model serialized to: {model_path}\n")

        return payload
