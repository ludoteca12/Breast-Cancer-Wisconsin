import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, TEST_SIZE  # importa seu random_state global


# =============================================================
# 1. Load Raw Data
# =============================================================
def load_raw_data(path: str) -> pd.DataFrame:
    """
    Loads the raw dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the raw dataset (CSV).

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(path)



# =============================================================
# 2. Initial Cleaning
# =============================================================
def initial_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies basic cleaning steps:
    - Removes ID-like columns
    - Drops unnamed columns
    - Ensures no unnecessary fields remain
    - Standarlaztion of column names

    Returns
    -------
    pd.DataFrame
    """
    
    rename_map = {
    "concave points_worst": "concave_points_worst",
    "concave points_se": "concave_points_se",
    "concave points_mean": "concave_points_mean",
    }
    
    df = df.rename(columns=rename_map)

    cols_to_drop = ['id', 'Unnamed: 32']
    df = df.drop(columns=cols_to_drop, errors="ignore")

    df = df.drop(columns=list(rename_map.keys()), errors="ignore")
    
    return df



# =============================================================
# 3. Encode Target
# =============================================================
def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the 'diagnosis' column from categorical (M/B)
    to numerical binary values (1/0).

    Returns
    -------
    pd.DataFrame
    """
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)
    return df



# =============================================================
# 4. Split Features and Target
# =============================================================
def split_features_target(df: pd.DataFrame, target: str = "diagnosis"):
    """
    Separates the dataset into X (features) and y (target).

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    X = df.drop(columns=[target])
    y = df[target]
    return X, y



# =============================================================
# 5. Train/Test Split
# =============================================================
def split_train_test(X, y):
    """
    Splits into training and test sets using stratification.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )



# =============================================================
# 6. Scaling (StandardScaler)
# =============================================================
def scale_features(X_train, X_test):
    """
    Applies StandardScaler to normalize numerical features.
    The scaler is fit ONLY on the training set to avoid data leakage.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  index=X_test.index,  columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, scaler



# =============================================================
# 7. Feature Aggregation (Variance Smoothing)
# =============================================================
def aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates aggregated / smoothed features by averaging related groups.
    Only uses the prefixes implemented by the user.

    This step helps reduce noise and variance among correlated features.

    Returns
    -------
    pd.DataFrame
    """

    prefixes = ["radius", "perimeter", "area", "concavity", "texture"]

    df = df.copy()

    for p in prefixes:
        group_cols = [col for col in df.columns if col.startswith(p)]
        if len(group_cols) > 1:
            df[f"{p}_avg"] = df[group_cols].mean(axis=1)

    df["var_total"] = df.var(axis=1)

    return df



# =============================================================
# 8. Save Processed Data
# =============================================================
def save_processed_data(X_train, X_test, y_train, y_test, folder="../../data/processed/"):
    """
    Saves the processed datasets to the processed folder.

    Parameters
    ----------
    folder : str
        Output directory.
    """

    X_train.to_csv(folder + "X_train_preprocessed.csv", index=False)
    X_test.to_csv(folder  + "X_test_preprocessed.csv",  index=False)
    y_train.to_csv(folder + "y_train.csv", index=False)
    y_test.to_csv(folder  + "y_test.csv",  index=False)

    print("Processed data saved successfully!")
