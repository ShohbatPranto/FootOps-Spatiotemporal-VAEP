import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from typing import Tuple, List, Dict, Any

from src.data.ingestion import EventIngestor
from src.data.spadl import SPADLConverter
from src.features.engineering import SpatialFeatureEngineer

class VAEPDatasetBuilder:
    """
    Constructs the dual targets (scores & concedes) for a pre-computed feature matrix.
    """
    def __init__(self, lookahead_actions: int = 10):
        self.lookahead = lookahead_actions

    def generate_targets(self, df_features: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculates ground-truth labels for VAEP:
        - Y_scores: 1 if the attacking team scores within the next N actions.
        - Y_concedes: 1 if the defending team scores within the next N actions.
        """
        print(f"🎯 Generating {self.lookahead}-action lookahead targets...")
        scores = pd.Series(0, index=df_features.index, dtype=np.int32)
        concedes = pd.Series(0, index=df_features.index, dtype=np.int32)

        for shift in range(self.lookahead):
            shifted_match = df_features['match_id'].shift(-shift)
            shifted_period = df_features['period'].shift(-shift)
            shifted_type = df_features['type'].shift(-shift)
            shifted_success = df_features['result_success'].shift(-shift)
            shifted_team = df_features['team'].shift(-shift)

            same_sequence = (shifted_match == df_features['match_id']) & (shifted_period == df_features['period'])
            is_shifted_goal = same_sequence & (shifted_type == 'Shot') & (shifted_success == 1)

            same_team_goal = is_shifted_goal & (shifted_team == df_features['team'])
            scores = scores | same_team_goal.astype(np.int32)

            opp_team_goal = is_shifted_goal & (shifted_team != df_features['team'])
            concedes = concedes | opp_team_goal.astype(np.int32)

        return scores, concedes

class VAEPModelTrainer:
    """
    Manages temporal train-validation-test splitting, training,
    and rigorous evaluation of dual LightGBM classifiers.
    """
    
    def __init__(self, categorical_features: List[str] = None):
        self.feature_cols = [
            'type', 'start_x', 'start_y', 'end_x', 'end_y',
            'start_dist_to_goal', 'start_angle_to_goal',
            'end_dist_to_goal', 'end_angle_to_goal', 'progression_dist', 'time_delta'
        ]
        self.categorical_features = categorical_features or ['type']
        self.model_scores = None
        self.model_concedes = None

    def temporal_split(self, df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[List[int], List[int]]:
        """
        Performs structural match-level splitting to prevent data leakage.
        Ensures actions from the same match are never split across train/test sets.
        """
        unique_matches = df['match_id'].unique()
        unique_matches = sorted(unique_matches)
        
        split_idx = int(len(unique_matches) * (1 - test_ratio))
        train_matches = unique_matches[:split_idx]
        test_matches = unique_matches[split_idx:]
        
        return list(train_matches), list(test_matches)

    def _prepare_lgb_dataset(self, df: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """Extracts feature columns and ensures correct data types for LightGBM."""
        X = df[self.feature_cols].copy()
        
        # Cast categorical columns to LightGBM category dtype
        for col in self.categorical_features:
            if col in X.columns:
                X[col] = X[col].astype('category')
                
        return X, self.categorical_features

    def train(self, df: pd.DataFrame, y_scores: pd.Series, y_concedes: pd.Series, test_ratio: float = 0.2) -> Dict[str, Any]:
        """Trains both P_scores and P_concedes LightGBM models."""
        
        # 1. Split matches to prevent leakages
        train_matches, test_matches = self.temporal_split(df, test_ratio)
        
        train_mask = df['match_id'].isin(train_matches)
        test_mask = df['match_id'].isin(test_matches)
        
        print(f"📊 Dataset Splits:")
        print(f"   Train: {sum(train_mask)} actions ({len(train_matches)} matches)")
        print(f"   Test:  {sum(test_mask)} actions ({len(test_matches)} matches)")

        # Prepare Features
        X_train, cat_cols = self._prepare_lgb_dataset(df[train_mask], y_scores[train_mask])
        X_test, _ = self._prepare_lgb_dataset(df[test_mask], y_scores[test_mask])

        # LightGBM Parameters optimized for spatial categorical configurations
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'random_state': 42
        }

        # Model A: Scoring Probability (P_scores)
        print("\n🔥 Training Model A: P_scores Classifier...")
        train_dataset = lgb.Dataset(X_train, label=y_scores[train_mask], categorical_feature=cat_cols)
        test_dataset = lgb.Dataset(X_test, label=y_scores[test_mask], reference=train_dataset, categorical_feature=cat_cols)
        
        # Train model with early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        self.model_scores = lgb.train(
            lgb_params,
            train_dataset,
            num_boost_round=300,
            valid_sets=[train_dataset, test_dataset],
            callbacks=callbacks
        )

        # Model B: Conceding Probability (P_concedes)
        print("🔥 Training Model B: P_concedes Classifier...")
        train_dataset_c = lgb.Dataset(X_train, label=y_concedes[train_mask], categorical_feature=cat_cols)
        test_dataset_c = lgb.Dataset(X_test, label=y_concedes[test_mask], reference=train_dataset_c, categorical_feature=cat_cols)
        
        self.model_concedes = lgb.train(
            lgb_params,
            train_dataset_c,
            num_boost_round=300,
            valid_sets=[train_dataset_c, test_dataset_c],
            callbacks=callbacks
        )

        # 2. Rigorous Evaluation
        y_pred_scores = self.model_scores.predict(X_test)
        y_pred_concedes = self.model_concedes.predict(X_test)
        
        auc_s = roc_auc_score(y_scores[test_mask], y_pred_scores)
        brier_s = brier_score_loss(y_scores[test_mask], y_pred_scores)
        
        auc_c = roc_auc_score(y_concedes[test_mask], y_pred_concedes)
        brier_c = brier_score_loss(y_concedes[test_mask], y_pred_concedes)

        metrics = {
            'p_scores_auc': auc_s,
            'p_scores_brier': brier_s,
            'p_concedes_auc': auc_c,
            'p_concedes_brier': brier_c
        }

        print("\n📈 --- DUAL MODEL METRICS ---")
        print(f"🚀 P_scores   - ROC-AUC: {auc_s:.4f} | Brier Score: {brier_s:.5f}")
        print(f"🛡️  P_concedes - ROC-AUC: {auc_c:.4f} | Brier Score: {brier_c:.5f}")
        
        return metrics

    def predict_probabilities(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Infers dual-probability profiles for a set of standardized spatial features."""
        if not self.model_scores or not self.model_concedes:
            raise RuntimeError("Models have not been trained yet.")
            
        X, _ = self._prepare_lgb_dataset(df, None)
        p_scores = self.model_scores.predict(X)
        p_concedes = self.model_concedes.predict(X)
        
        return p_scores, p_concedes