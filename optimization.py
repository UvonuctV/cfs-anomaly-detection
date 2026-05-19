"""
optimization.py — Feature and Sensor Selection Module
=====================================================
Містить методи відбору ознак для зменшення розмірності простору 
та виділення найбільш критичних сенсорів в ICS (BATADAL, WADI, HAI).

Implemented methods based on 2024-2025 research:
1. Random Forest Feature Importance (Embedded method)
2. Mutual Information (Filter method)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

def apply_rf_importance(train_df, test_df, sensors, label_col, top_k=10, use_smote=True):
    """
    Select top_k sensors using Random Forest Feature Importance.
    Uses test_df (which contains attack labels) to find importance, 
    as train_df usually contains only normal data in these datasets.
    """
    print(f"  [Optimization] Running Random Forest Feature Importance (Top {top_k})...")
    
    # We need both normal and attack data to find importance.
    # We will use the test set as it contains both (or combine them).
    X = test_df[sensors].fillna(0)
    y = test_df[label_col].fillna(0).astype(int)
    
    if len(np.unique(y)) < 2:
        print("  [Optimization] WARNING: Only one class found in test set. Cannot compute importance. Retaining all sensors.")
        return sensors
    
    # Optionally apply SMOTE to balance the classes for better feature selection
    if use_smote and y.sum() > 5:
        try:
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            X_train, y_train = X_res, y_res
        except Exception as e:
            print(f"  [Optimization] SMOTE failed ({e}). Using raw data.")
            X_train, y_train = X, y
    else:
         X_train, y_train = X, y

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    importance = clf.feature_importances_
    sensor_importance = pd.Series(importance, index=sensors).sort_values(ascending=False)
    
    print("\n  Top Sensors by RF Importance:")
    for sensor, imp in sensor_importance.head(top_k).items():
        print(f"    - {sensor}: {imp:.4f}")
        
    return sensor_importance.head(top_k).index.tolist()

def apply_mutual_info(train_df, test_df, sensors, label_col, top_k=10):
    """
    Select top_k sensors using Mutual Information statistics.
    """
    print(f"  [Optimization] Running Mutual Information Feature Selection (Top {top_k})...")
    
    X = test_df[sensors].fillna(0)
    y = test_df[label_col].fillna(0).astype(int)
    
    if len(np.unique(y)) < 2:
        print("  [Optimization] WARNING: Only one class found in test set. Cannot compute MI. Retaining all sensors.")
        return sensors
        
    # sample for speed if dataset is huge
    if len(X) > 50000:
        idx = np.random.choice(len(X), 50000, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
    else:
        X_sample = X
        y_sample = y
        
    mi_scores = mutual_info_classif(X_sample, y_sample, discrete_features=False, random_state=42)
    sensor_mi = pd.Series(mi_scores, index=sensors).sort_values(ascending=False)
    
    print("\n  Top Sensors by Mutual Information:")
    for sensor, mi in sensor_mi.head(top_k).items():
        print(f"    - {sensor}: {mi:.4f}")
        
    return sensor_mi.head(top_k).index.tolist()
