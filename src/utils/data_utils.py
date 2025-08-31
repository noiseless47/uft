"""
Data utility functions for the football squad selection pipeline.
"""

import pandas as pd
import numpy as np
import hashlib
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def validate_schema(df: pd.DataFrame, required_columns: List[str], 
                   table_name: str = "unknown") -> bool:
    """Validate DataFrame schema against required columns."""
    missing_cols = set(required_columns) - set(df.columns)
    
    if missing_cols:
        logger.error(f"{table_name}: Missing columns: {missing_cols}")
        return False
    
    logger.info(f"{table_name}: Schema validation passed")
    return True


def compute_checksum(filepath: str) -> str:
    """Compute SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    
    return sha256_hash.hexdigest()


def check_data_quality(df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
    """Check data quality metrics."""
    quality_report = {
        'table_name': table_name,
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_data_rate': df.isnull().sum().sum() / (len(df) * len(df.columns)),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    # Column-specific checks
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        quality_report['outliers_3sigma'] = (
            np.abs(df[numeric_cols] - df[numeric_cols].mean()) > 
            3 * df[numeric_cols].std()
        ).sum().sum()
    
    return quality_report


def create_temporal_splits(df: pd.DataFrame, date_column: str, 
                          n_splits: int = 5, gap_days: int = 7,
                          test_size_days: int = 30) -> List[Dict[str, Any]]:
    """Create temporal cross-validation splits."""
    df_sorted = df.sort_values(date_column)
    dates = pd.to_datetime(df_sorted[date_column])
    
    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days
    
    splits = []
    
    for i in range(n_splits):
        # Calculate split boundaries
        test_end = max_date - timedelta(days=i * (test_size_days + gap_days))
        test_start = test_end - timedelta(days=test_size_days)
        train_end = test_start - timedelta(days=gap_days)
        
        # Create boolean masks
        train_mask = dates <= train_end
        test_mask = (dates >= test_start) & (dates <= test_end)
        
        splits.append({
            'fold': i,
            'train_indices': df_sorted[train_mask].index.tolist(),
            'test_indices': df_sorted[test_mask].index.tolist(),
            'train_start': min_date,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'n_train': train_mask.sum(),
            'n_test': test_mask.sum()
        })
    
    return splits


def impute_missing_values(df: pd.DataFrame, strategy: str = "median",
                         group_by: Optional[str] = None) -> pd.DataFrame:
    """Impute missing values with specified strategy."""
    df_imputed = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().any():
            if group_by and group_by in df.columns:
                # Group-based imputation
                if strategy == "median":
                    df_imputed[col] = df.groupby(group_by)[col].transform(
                        lambda x: x.fillna(x.median())
                    )
                elif strategy == "mean":
                    df_imputed[col] = df.groupby(group_by)[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
            else:
                # Global imputation
                if strategy == "median":
                    df_imputed[col] = df[col].fillna(df[col].median())
                elif strategy == "mean":
                    df_imputed[col] = df[col].fillna(df[col].mean())
                elif strategy == "zero":
                    df_imputed[col] = df[col].fillna(0)
    
    return df_imputed


def detect_outliers(df: pd.DataFrame, method: str = "iqr", 
                   threshold: float = 3.0) -> pd.DataFrame:
    """Detect outliers in numerical columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_mask = pd.DataFrame(False, index=df.index, columns=numeric_cols)
    
    for col in numeric_cols:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask[col] = z_scores > threshold
    
    return outlier_mask


def create_feature_windows(df: pd.DataFrame, player_col: str, date_col: str,
                          value_cols: List[str], windows: List[int]) -> pd.DataFrame:
    """Create rolling window features for time series data."""
    df_sorted = df.sort_values([player_col, date_col])
    
    windowed_features = []
    
    for window in windows:
        for value_col in value_cols:
            # Calculate rolling statistics
            rolling_mean = df_sorted.groupby(player_col)[value_col].rolling(
                window=window, min_periods=1
            ).mean().reset_index(level=0, drop=True)
            
            rolling_std = df_sorted.groupby(player_col)[value_col].rolling(
                window=window, min_periods=1
            ).std().reset_index(level=0, drop=True)
            
            # Add to DataFrame
            df_sorted[f"{value_col}_mean_{window}"] = rolling_mean
            df_sorted[f"{value_col}_std_{window}"] = rolling_std.fillna(0)
    
    return df_sorted
