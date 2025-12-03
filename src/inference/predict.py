# src/inference/predict.py

import joblib
import pandas as pd
import numpy as np
from pathlib import Path


class PredictionPipeline:
    """
    A reusable prediction pipeline that:
        - Loads the trained model bundle
        - Validates incoming data
        - Ensures correct feature ordering
        - Generates predictions and probabilities

    Usage:
        pipeline = PredictionPipeline()
        result = pipeline.predict_single(sample_dict)
    """

    def __init__(self, bundle_path="../src/models/final_model.pkl"):
        self.bundle_path = Path(bundle_path)
        self._load_bundle()

    # -----------------------------------------------------------
    # 1. LOAD MODEL BUNDLE (model, scaler, selected features)
    # -----------------------------------------------------------
    def _load_bundle(self):
        """Load the trained model, scaler and selected features."""
        if not self.bundle_path.exists():
            raise FileNotFoundError(f"Model bundle not found at: {self.bundle_path}")

        bundle = joblib.load(self.bundle_path)

        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.features = list(bundle["features"])

    # -----------------------------------------------------------
    # 2. VALIDATE AND PREPARE INPUT
    # -----------------------------------------------------------
    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures input data contains all required features in the correct order.
        Extra columns are ignored. Missing columns trigger an error.
        """

        missing = [col for col in self.features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Select and order only the required features
        df = df.loc[:, self.features]

        return df

    # -----------------------------------------------------------
    # 3. PREDICT A SINGLE SAMPLE (dict → DataFrame → prediction)
    # -----------------------------------------------------------
    def predict_single(self, sample: dict) -> pd.DataFrame:
        """
        Predict a single new sample represented as a Python dictionary.
        Returns a DataFrame combining input features + prediction output.
        """
        df_input = pd.DataFrame([sample])
        df_prepared = self._validate_and_prepare(df_input)
    
        pred = self.model.predict(df_prepared)[0]
        prob = self.model.predict_proba(df_prepared)[0, 1]
    
        # Build output dataframe
        df_output = df_input.copy()
        df_output["prediction"] = int(pred)
        df_output["prediction_label"] = "Malignant" if pred == 1 else "Benign"
        df_output["probability_malignant"] = float(prob)

        return df_output

    # -----------------------------------------------------------
    # 4. PREDICT A BATCH OF SAMPLES
    # -----------------------------------------------------------
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        df_prepared = self._validate_and_prepare(df)
    
        preds = self.model.predict(df_prepared)
        probs = self.model.predict_proba(df_prepared)[:, 1]
    
        df_output = df.copy()
        df_output["prediction"] = preds
        df_output["prediction_label"] = np.where(preds == 1, "Malignant", "Benign")
        df_output["probability_malignant"] = probs
    
        return df_output

