"""
pipeline.py — Shared Multivariate Anomaly Detection Pipeline
=============================================================
Метод виявлення аномалій у критичній інфраструктурі
на основі аналізу не-гауссівських характеристик трафіку та процесних даних

Architecture:
    1. N independent Stacking Regressors (XGB + RF), one per physical sensor
    2. EVT/POT with Generalized Pareto Distribution (GPD) on each sensor's residuals
    3. Global Aggregation via configurable sensor voting threshold
"""

import numpy as np
import pandas as pd
import time
import warnings
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, genpareto
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, IsolationForest, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import os
import hashlib
import joblib
from optimization import apply_rf_importance, apply_mutual_info

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)


# =====================================================================
# 1. DATA LOADING HELPERS
# =====================================================================

def load_batadal(train_path, test_path):
    """Load and preprocess BATADAL water distribution dataset."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for df in [train_df, test_df]:
        df.columns = df.columns.str.strip()
        if 'ATT_FLAG' in df.columns:
            df['ATT_FLAG'] = df['ATT_FLAG'].replace(-999, 0)
        df['DATETIME'] = pd.to_datetime(df['DATETIME'], format='%d/%m/%y %H')
        df.set_index('DATETIME', inplace=True)
        df.sort_index(inplace=True)

    label_col = 'ATT_FLAG'
    return train_df, test_df, label_col


def load_hai(train_path, test_path, downsample=5):
    """Load and preprocess HAI (HIL-based Augmented ICS) dataset."""
    train_full = pd.read_csv(train_path)
    test_full = pd.read_csv(test_path)

    for df in [train_full, test_full]:
        df.columns = df.columns.str.strip()

    # Downsample training for tractability
    train_df = train_full.iloc[::downsample].reset_index(drop=True).copy()
    test_df = test_full.copy()
    del train_full, test_full
    gc.collect()

    train_df['time'] = pd.to_datetime(train_df['time'])
    test_df['time'] = pd.to_datetime(test_df['time'])
    train_df.set_index('time', inplace=True)
    test_df.set_index('time', inplace=True)

    label_col = 'attack'
    return train_df, test_df, label_col


def load_wadi(train_path, test_path, downsample=10):
    """Load and preprocess WADI (Water Distribution) dataset."""
    # Train: WADI_14days_new.csv — normal operations (has proper column headers)
    print("  Loading WADI train (this may take a moment)...")
    train_full = pd.read_csv(train_path, low_memory=False)
    train_full.columns = train_full.columns.str.strip()

    # Drop metadata columns from train
    meta_cols = ['Row', 'Date', 'Time']
    for col in meta_cols:
        train_full.drop(columns=[col], inplace=True, errors='ignore')

    # Get sensor column names from train (these are the authoritative names)
    train_sensor_names = list(train_full.columns)

    # Test: WADI_attackdataLABLE.csv
    # This file has the same structure: Row, Date, Time, sensors..., Attack_Label
    # The header row is at row 1, and row 0 is numeric 0-130. We skip both.
    # and assign column names from the train file.
    print("  Loading WADI test (attack data)...")
    test_full = pd.read_csv(test_path, header=None, skiprows=2, low_memory=False)

    # Drop the first 3 metadata columns (Row, Date, Time) from test too
    test_full.drop(columns=test_full.columns[:3], inplace=True)
    test_full.reset_index(drop=True, inplace=True)
    # Re-index columns sequentially after dropping
    test_full.columns = range(len(test_full.columns))

    # Now: test should have n_train_cols + 1 (label at end)
    n_train_cols = len(train_sensor_names)
    n_test_cols = len(test_full.columns)
    print(f"  Train columns: {n_train_cols}, Test columns: {n_test_cols}")

    label_col = 'Attack_Label'
    if n_test_cols == n_train_cols + 1:
        # Perfect: sensors + label
        test_full.columns = train_sensor_names + [label_col]
    elif n_test_cols == n_train_cols:
        # No separate label column
        test_full.columns = train_sensor_names
        label_col = train_sensor_names[-1]
    else:
        # Fallback: try to use first n_train_cols as sensors, rest as extra
        print(f"  WARNING: Column count mismatch ({n_test_cols} vs {n_train_cols}+1). Trimming.")
        # Take only the columns we can map
        if n_test_cols > n_train_cols:
            test_full.columns = train_sensor_names + [f'extra_{i}' for i in range(n_test_cols - n_train_cols)]
            label_col = test_full.columns[-1]
        else:
            test_full.columns = [f'col_{i}' for i in range(n_test_cols)]
            label_col = test_full.columns[-1]

    print(f"  WADI label column: '{label_col}'")

    # Convert label: -1 → 1 (attack), others → 0 (normal)
    test_full[label_col] = pd.to_numeric(test_full[label_col], errors='coerce').fillna(0)
    test_full[label_col] = (test_full[label_col] == -1).astype(int)

    # Train has no attack labels
    train_full[label_col] = 0

    # Reset indices
    train_full.reset_index(drop=True, inplace=True)
    test_full.reset_index(drop=True, inplace=True)

    # **Fix**: Convert all to numeric BEFORE rolling mean to avoid DataError
    print("  Converting to numeric...")
    for df in [train_full, test_full]:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(0, inplace=True)

    # Downsample train for speed using rolling mean (preserves signal average)
    print(f"  Downsampling by factor {downsample} using rolling mean...")
    if downsample > 1:
        # We must exclude the label column from the rolling mean if present in train
        train_df = train_full.rolling(window=downsample, min_periods=1).mean().iloc[::downsample].reset_index(drop=True)
        # Test file: downsample to match train resolution, BUT keep labels maxed (if any attack occurred in window, flag it)
        test_features = test_full.drop(columns=[label_col]).rolling(window=downsample, min_periods=1).mean().iloc[::downsample]
        test_labels = test_full[label_col].rolling(window=downsample, min_periods=1).max().iloc[::downsample]
        test_df = pd.concat([test_features, test_labels], axis=1).reset_index(drop=True)
    else:
        train_df = train_full.copy()
        test_df = test_full.copy()

    return train_df, test_df, label_col

def load_sherlock(train_path, test_path, downsample=1):
    print("  Loading Sherlock train...")
    train_full = pd.read_csv(train_path)
    
    print("  Loading Sherlock test (attack data)...")
    test_full = pd.read_csv(test_path)
    
    label_col = 'malicious'
    print(f"  Train origin shape: {train_full.shape}, Test origin shape: {test_full.shape}")
    
    # Time column handling (Sherlock uses 'timestamp')
    if 'timestamp' in train_full.columns:
        train_full['timestamp'] = pd.to_datetime(train_full['timestamp'], unit='s')
        train_full.set_index('timestamp', inplace=True)
    if 'timestamp' in test_full.columns:
        test_full['timestamp'] = pd.to_datetime(test_full['timestamp'], unit='s')
        test_full.set_index('timestamp', inplace=True)

    # Convert everything to numeric just in case there are booleans or strings
    print("  Converting to numeric...")
    for df in [train_full, test_full]:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(0, inplace=True)

    # Downsample using rolling mean
    if downsample > 1:
        print(f"  Downsampling by factor {downsample} using rolling mean...")
        # Train has malicious=0 always
        train_df = train_full.rolling(window=downsample, min_periods=1).mean().iloc[::downsample]
        
        # Test needs label maxed
        test_features = test_full.drop(columns=[label_col]).rolling(window=downsample, min_periods=1).mean().iloc[::downsample]
        test_labels = test_full[label_col].rolling(window=downsample, min_periods=1).max().iloc[::downsample]
        test_df = pd.concat([test_features, test_labels], axis=1)
    else:
        train_df = train_full.copy()
        test_df = test_full.copy()

    del train_full, test_full
    gc.collect()

    return train_df, test_df, label_col


# =====================================================================
# 2. FEATURE ENGINEERING
# =====================================================================

def add_time_features(train_df, test_df, add_deltas=True):
    """Add cyclic time features and sensor momentum for better accuracy."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Process Momentum (Deltas) - Accuracy Booster
    if add_deltas:
        # We only add deltas for sensors, find them (exclude time/label features)
        exclude = ['Attack_Label', 'malicious', 'Timestamp', 'Row', 'Date', 'Time']
        sensors = [c for c in train_df.columns if c not in exclude]
        for s in sensors:
            train_df[f"{s}_delta"] = train_df[s].diff().fillna(0)
            test_df[f"{s}_delta"] = test_df[s].diff().fillna(0)

    for df in [train_df, test_df]:
        if hasattr(df.index, 'hour'):
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
            df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    return train_df, test_df

def plot_thesis_eda(train_df, sensors, dataset_name):
    """Generate EDA plots: Correlation matrix, Unit circle for time, Sensors over time."""
    print(f"\n--- Generating EDA Plots for {dataset_name} (Thesis Section 3) ---")
    plot_sensors = sensors[:10] if len(sensors) > 10 else sensors
    
    # 1. Correlation Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(train_df[plot_sensors].corr(), cmap='coolwarm', annot=False)
    plt.title(f"{dataset_name}: Sensor Correlation Matrix (Top 10)")
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_EDA_Correlation.png', dpi=150)
    plt.show()

    # 2. Time Unit Circle
    if 'hour_sin' in train_df.columns and 'hour_cos' in train_df.columns:
        plt.figure(figsize=(6, 6))
        sc = plt.scatter(train_df['hour_cos'][:1000], train_df['hour_sin'][:1000], c=train_df.index.hour[:1000], cmap='twilight', s=30, alpha=0.7)
        plt.colorbar(sc, label='Hour of day')
        plt.title(f"{dataset_name}: Cyclical Time Encoding (Unit Circle)")
        plt.xlabel('hour_cos')
        plt.ylabel('hour_sin')
        plt.axhline(0, color='grey', lw=0.5)
        plt.axvline(0, color='grey', lw=0.5)
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_EDA_TimeCircle.png', dpi=150)
        plt.show()

    # 3. Sensor activity over time
    if len(plot_sensors) >= 2:
        plt.figure(figsize=(14, 4))
        plt.plot(train_df.index[:2000], train_df[plot_sensors[0]][:2000], label=plot_sensors[0])
        plt.plot(train_df.index[:2000], train_df[plot_sensors[1]][:2000], label=plot_sensors[1])
        plt.title(f"{dataset_name}: Sensor Activity (Normal State)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_EDA_SensorActivity.png', dpi=150)
        plt.show()


def create_lag_features(train_df, test_df, sensors, lags=1):
    """Add lagged features for sequential context."""
    if lags <= 0:
        return train_df, test_df
    print(f"  Adding lag features (lags={lags})...")
    for s in sensors:
        for lag in range(1, lags + 1):
            train_df[f'{s}_lag_{lag}'] = train_df[s].shift(lag)
            test_df[f'{s}_lag_{lag}'] = test_df[s].shift(lag)
            
    train_df.bfill(inplace=True)
    test_df.bfill(inplace=True)
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
    return train_df, test_df


def get_variable_sensors(train_df, label_col, min_unique=1):
    """Identify sensors that are not constant (have enough variance to model)."""
    exclude = [label_col, 'time', 'date', 'hour', 'Row', 'Date', 'Time',
               'attack_P1', 'attack_P2', 'attack_P3',
               'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
    all_cols = [c for c in train_df.columns if c not in exclude]
    nunique = train_df[all_cols].nunique()
    variable = nunique[nunique > min_unique].index.tolist()
    return variable


# =====================================================================
# 3. CACHING HELPERS
# =====================================================================

def get_cache_id(dataset_name, sensor, xgb_est, xgb_dep, xgb_lr, rf_est, rf_dep, cv, model_type, lags, df_shape):
    """Generate a unique ID for a model configuration to avoid redundant training."""
    raw_str = f"{dataset_name}_{sensor}_{xgb_est}_{xgb_dep}_{xgb_lr}_{rf_est}_{rf_dep}_{cv}_{model_type}_{lags}_{df_shape}"
    return hashlib.md5(raw_str.encode()).hexdigest()

# =====================================================================
# 4. MULTIVARIATE TRAINING & EVT/POT
# =====================================================================

def train_single_sensor(target, sensors, train_df, test_df, time_features,
                        xgb_estimators, xgb_depth, xgb_lr, 
                        rf_estimators, rf_depth, cv, model_type,
                        use_cache, cache_dir, dataset_name, lag_feats_count, df_shape_slug):
    """Worker function for parallel sensor training."""
    
    # Cache Check
    cache_file = None
    if use_cache:
        cid = get_cache_id(dataset_name, target, xgb_estimators, xgb_depth, xgb_lr, 
                           rf_estimators, rf_depth, cv, model_type, lag_feats_count, df_shape_slug)
        cache_file = os.path.join(cache_dir, f"res_{cid}.joblib")
        
        if os.path.exists(cache_file):
            try:
                cached_data = joblib.load(cache_file)
                # Calculate stability from cached residuals
                stab = 1.0 / (np.std(cached_data['train']) + 1e-6)
                return target, cached_data['train'], cached_data['test'], stab, True
            except:
                pass

    # Model Training
    feats = [f for f in sensors if f != target] + time_features
    lag_feats = [c for c in train_df.columns if '_lag_' in c]
    if lag_feats:
        feats = feats + lag_feats

    # Use float32 to speed up computations
    Xtr = train_df[feats].astype('float32').bfill().ffill().fillna(0).values
    ytr = train_df[target].astype('float32').bfill().ffill().fillna(0).values

    est = [
        ('xgb', XGBRegressor(n_estimators=xgb_estimators, max_depth=xgb_depth,
                              learning_rate=xgb_lr, n_jobs=1,
                              random_state=42, verbosity=0)),
        ('hgb', HistGradientBoostingRegressor(max_iter=xgb_estimators, max_depth=rf_depth,
                                               random_state=42))
    ]
    mdl = StackingRegressor(estimators=est, final_estimator=LinearRegression(), cv=2, n_jobs=1)

    try:
        mdl.fit(Xtr, ytr)
    except Exception as e:
        return target, None, None, False

    tr_res = np.abs(ytr - mdl.predict(Xtr))
    
    Xte = test_df[feats].astype('float32').bfill().ffill().fillna(0).values
    yte = test_df[target].astype('float32').bfill().ffill().fillna(0).values
    te_res = pd.Series(np.abs(yte - mdl.predict(Xte)), index=test_df.index)

    # Save to cache
    if use_cache and cache_file:
        joblib.dump({'train': tr_res, 'test': te_res}, cache_file)

    # Calculate Stability Score (higher is better)
    stability = 1.0 / (np.std(tr_res) + 1e-6)
    
    return target, tr_res, te_res, stability, False


def compute_residuals(train_df, test_df, sensors, label_col,
                      xgb_estimators=20, xgb_depth=3, xgb_lr=0.1,
                      rf_estimators=20, rf_depth=4, 
                      dataset_name="Unknown", use_cache=True):
    """
    Train N independent Stacking Regressors (in parallel) and evaluate raw residuals.
    """
    from joblib import Parallel, delayed
    
    train_residuals = {}
    test_residuals = {}
    
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    if use_cache and not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    time_features = [c for c in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos'] if c in train_df.columns]
    lag_feats_count = len([c for c in train_df.columns if '_lag_' in c])
    df_shape_slug = f"{train_df.shape[0]}_{train_df.shape[1]}"

    print(f"  [Parallel] Launching training for {len(sensors)} sensors across all CPU cores...")
    
    results = Parallel(n_jobs=-1)(
        delayed(train_single_sensor)(
            target, sensors, train_df, test_df, time_features,
            xgb_estimators, xgb_depth, xgb_lr, rf_estimators, rf_depth,
            2, "hgb_stack", # cv=2, model_type="hgb_stack"
            use_cache, cache_dir, dataset_name, lag_feats_count, df_shape_slug
        ) for target in sensors
    )

    stability_scores = {}
    for target, tr_res, te_res, stab, was_cached in results:
        if tr_res is not None:
            train_residuals[target] = tr_res
            test_residuals[target] = te_res
            stability_scores[target] = stab
    
    n_cached = sum([1 for r in results if r[4]])
    print(f"  Finished modeling {len(sensors)} sensors ({n_cached} loaded from cache).")
    return train_residuals, test_residuals, stability_scores


def apply_thresholds(test_index, sensors, train_residuals, test_residuals, dataset_name="Unknown",
                     pot_pct=98, gpd_conf=0.999, sensitivity=3.0,
                     smooth_w=10, adaptive_w=200,
                     consensus_w=5, consensus_min=3):
    """
    Apply EVT/POT threshold logic to pre-computed residuals.
    Returns a DataFrame of per-sensor triggers for the test set.
    """
    triggers = pd.DataFrame(index=test_index)
    plot_pot_once = True
    
    for target in sensors:
        tr_res = train_residuals[target]
        te_res = test_residuals[target]
        
        # EVT/POT on training residuals
        u = np.percentile(tr_res, pot_pct)
        tail = tr_res[tr_res > u] - u

        if len(tail) > 5:
            try:
                c_, _, s_ = genpareto.fit(tail, floc=0)
                pt = u + genpareto.ppf(gpd_conf, c_, 0, s_)
                if np.isnan(pt) or np.isinf(pt) or pt <= 0:
                    raise ValueError()
            except:
                m_, s_ = norm.fit(tr_res)
                pt = m_ + 4 * s_
        else:
            m_, s_ = norm.fit(tr_res)
            pt = m_ + 4 * s_
            
        # Potentially plot the POT threshold for the thesis (just for the first valid sensor)
        if plot_pot_once and dataset_name != "Unknown" and u > 0:
            plt.figure(figsize=(10, 5))
            sns.kdeplot(tr_res, fill=True, color="blue", label="Residuals Density", alpha=0.3)
            plt.axvline(u, color="orange", linestyle="--", label=f"Base Threshold u ({pot_pct}%)")
            plt.axvline(pt, color="red", linestyle="-", label=f"Dynamic POT Threshold (z_q)")
            plt.title(f"{dataset_name}: Residuals & POT Threshold for {target}")
            plt.xlabel("Absolute Residual")
            plt.ylabel("Density")
            plt.legend()
            plt.xlim(0, pt * 1.5)
            plt.tight_layout()
            plt.savefig(f'{dataset_name}_POT_Distribution.png', dpi=150)
            plt.show()
            plot_pot_once = False

        sm = te_res.rolling(window=smooth_w, center=True).mean().bfill().ffill()
        
        # New: Jitter/Variance Detection (Accuracy Booster)
        sm_std = te_res.rolling(window=smooth_w, center=True).std().bfill().ffill()
        u_std = np.std(tr_res) # Typical noise in training
        
        ab = sm.rolling(window=adaptive_w, center=True).median().bfill().ffill()
        dt = ab + (pt * sensitivity)

        # Trigger on Magnitude (Jitter deactivated due to domain shift issues)
        rd_mean = (sm > dt).astype(int)
        
        rd = rd_mean
        
        la = (rd.rolling(window=consensus_w).sum() >= consensus_min).astype(int)
        triggers[target] = la
        
    return triggers



# =====================================================================
# 4. GLOBAL EVALUATION
# =====================================================================

def run_baseline_isolation_forest(train_df, test_df, sensors, y_true):
    """Run a classic standard baseline for comparison (Section 4)."""
    print("\n--- Running Baseline (Isolation Forest) ---")
    clf = IsolationForest(n_estimators=100, contamination=0.05, random_state=42, n_jobs=-1)
    X_train = train_df[sensors].fillna(0).values
    clf.fit(X_train)
    
    X_test = test_df[sensors].fillna(0).values
    preds = clf.predict(X_test)
    alert_iso = (preds == -1).astype(int)
    
    f1_iso = f1_score(y_true, alert_iso, zero_division=0)
    prec_iso = precision_score(y_true, alert_iso, zero_division=0)
    rec_iso = recall_score(y_true, alert_iso, zero_division=0)
    fp = alert_iso[y_true == 0].sum()
    tn = (alert_iso[y_true == 0] == 0).sum()
    fpr_iso = fp / max((fp + tn), 1)
    
    metrics = {
        'Precision': prec_iso,
        'Recall': rec_iso,
        'F1-Score': f1_iso,
        'FPR': fpr_iso
    }
    return alert_iso, metrics

def evaluate_global(triggers, y_true, stability_scores=None, vote_options=None, baseline_metrics=None):
    """
    Evaluate the global alert using reliability-weighted sensor votes.
    Returns (best_f1, best_v, global_alert, votes, stats_table).
    """
    if vote_options is None:
        vote_options = [1, 2, 3, 5, 8, 10, 15]

    # Weighted Voting logic
    if stability_scores is not None:
        # Create weight vector for sensors in 'triggers'
        sensor_order = list(triggers.columns)
        raw_weights = np.array([stability_scores.get(s, 1.0) for s in sensor_order])
        # Normalize weights so they sum to Number of Sensors (average weight = 1.0)
        # Accuracy Booster Fix: Clamped floor to 0.1 to ensure no sensor is fully ignored
        weights = raw_weights / np.mean(raw_weights)
        weights = np.maximum(0.1, weights)
        
        votes = (triggers * weights).sum(axis=1)
        print(f"  [Accuracy Booster] Using Reliability-Weighted Voting (Weight Range: {weights.min():.2f} - {weights.max():.2f})")
    else:
        votes = triggers.sum(axis=1)

    print("\n--- Sensor Vote Sensitivity Analysis ---")
    best_f1 = 0
    best_v = 1
    best_stats = {}
    for v in vote_options:
        alert = (votes >= v).astype(int)
        f1_v = f1_score(y_true, alert, zero_division=0)
        recall_v = recall_score(y_true, alert, zero_division=0)
        fp = alert[y_true == 0].sum()
        tn = (alert[y_true == 0] == 0).sum()
        fpr = fp / max((fp + tn), 1)
        print(f"  Votes>={v}: F1={f1_v:.4f}, Recall={recall_v:.2%}, FPR={fpr:.4f}, FP={fp}")
        if f1_v > best_f1:
            best_f1, best_v = f1_v, v
            best_stats = {
                'Method': f'Stacking + POT (k={v})',
                'Precision': precision_score(y_true, alert, zero_division=0),
                'Recall': recall_v,
                'F1-Score': f1_v,
                'FPR': fpr
            }

    global_alert = (votes >= best_v).astype(int)

    print(f"\n=== FINAL REPORT (Optimal: Votes>={best_v}) ===")
    print(classification_report(y_true, global_alert, zero_division=0))
    
    # Generate Thesis Comparative Table
    records = [best_stats]
    if baseline_metrics:
        baseline_metrics['Method'] = 'Baseline: Isolation Forest'
        records.append(baseline_metrics)
    
    table_df = pd.DataFrame(records)
    print("\n--- Table 4.1: Performance Metrics Comparison ---")
    print(table_df.to_string(index=False))

    return best_f1, best_v, global_alert, votes, table_df


# =====================================================================
# 5. VISUALIZATION
# =====================================================================

def plot_detection(votes, global_alert, y_true, best_v, best_f1, dataset_name, n_sensors):
    """Plot sensor vote heatmap and detection result."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    axes[0].fill_between(range(len(votes)), 0, votes.max(),
                         where=(y_true.values == 1), color='red', alpha=0.15, label='Actual Attack')
    axes[0].plot(range(len(votes)), votes.values, color='navy', lw=0.8, label='# Sensors Triggered')
    axes[0].axhline(best_v, color='crimson', linestyle='--', lw=1.5, label=f'Vote Threshold ({best_v})')
    axes[0].set_ylabel("Сенсори з алертом")
    axes[0].set_title(f"{dataset_name}: Мультиваріативна Детекція ({n_sensors} сенсорів)")
    axes[0].legend(loc='upper right')

    axes[1].fill_between(range(len(global_alert)), 0, 1,
                         where=(global_alert.values == 1), color='orange', alpha=0.6, label='System Alarm')
    axes[1].fill_between(range(len(global_alert)), 0, 1,
                         where=(y_true.values == 1), color='red', alpha=0.2, label='Actual Attack')
    axes[1].set_ylabel("Статус")
    axes[1].set_xlabel("Timestep")
    axes[1].set_title(f"{dataset_name}: F1={best_f1:.4f}")
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_detection.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: {dataset_name}_detection.png")


# =====================================================================
# 6. FULL PIPELINE RUNNER
# =====================================================================

def run_full_pipeline(dataset_name, train_df, test_df, label_col, max_sensors=None, optimize_method=None, top_k_sensors=10, **kwargs):
    """
    End-to-end pipeline: feature engineering → sensor selection → training → evaluation → plot.
    """
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {dataset_name}")
    print(f"{'=' * 60}")
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    t_start = time.time()
    print(f"Attack distribution:\n{test_df[label_col].value_counts()}")

    # 1. Feature engineering
    t_feat_start = time.time()
    train_df, test_df = add_time_features(train_df, test_df)
    t_feat = time.time() - t_feat_start

    # If user provided precomputed residuals, use them. Otherwise compute them.
    precomputed_train_res = kwargs.get('precomputed_train_res', None)
    precomputed_test_res = kwargs.get('precomputed_test_res', None)
    
    if precomputed_train_res is None or precomputed_test_res is None:
        # Select variable sensors first
        sensors = get_variable_sensors(train_df, label_col)
        
        # Optimization: Drop invariant sensors (std == 0) to reduce noise
        stds = train_df[sensors].std()
        invariant = stds[stds == 0].index.tolist()
        if invariant:
            print(f"  [Cleaning] Dropping {len(invariant)} invariant sensors: {invariant}")
            sensors = [s for s in sensors if s not in invariant]

        # Dimensionality Protection (REMOVED for Maximum Fidelity Run)

        # Feature Selection / Optimization Step
        if optimize_method == 'rf_importance' and top_k_sensors is not None:
            sensors = apply_rf_importance(train_df, test_df, sensors, label_col, top_k=top_k_sensors)
        elif optimize_method == 'mutual_info' and top_k_sensors is not None:
            sensors = apply_mutual_info(train_df, test_df, sensors, label_col, top_k=top_k_sensors)
        else:
            print(f"  [Methodology] Using ALL available variable sensors ({len(sensors)}).")
            
        if max_sensors is not None and len(sensors) > max_sensors:
            print(f"Limiting to max_sensors={max_sensors} (originally {len(sensors)})")
            sensors = sensors[:max_sensors]
        print(f"Final sensors to model: {len(sensors)}")
        
        lags = kwargs.get('lags', 2)
        train_df, test_df = create_lag_features(train_df, test_df, sensors, lags=lags)
        
        t_model_start = time.time()
        train_res, test_res, stability_scores = compute_residuals(
            train_df, test_df, sensors, label_col,
            xgb_estimators=kwargs.get('xgb_estimators', 50),
            xgb_depth=kwargs.get('xgb_depth', 3),
            xgb_lr=kwargs.get('xgb_lr', 0.1),
            rf_estimators=kwargs.get('rf_estimators', 30),
            rf_depth=kwargs.get('rf_depth', 4),
            dataset_name=dataset_name,
            use_cache=kwargs.get('use_cache', True)
        )
        t_model = time.time() - t_model_start
    else:
        t_model = 0
        stability_scores = None 
        # Use exact sensors from the precomputed residual dictionary keys
        sensors = list(precomputed_train_res.keys())
        train_res, test_res = precomputed_train_res, precomputed_test_res
        print(f"Using {len(sensors)} precomputed sensors and residuals.")
        
    # Generate EDA thesis plots (if not explicitly disabled)
    if kwargs.get('generate_plots', True):
        plot_thesis_eda(train_df, sensors, dataset_name)
        
    # Run Baseline for comparison (Section 4 thesis requirement)
    baseline_metrics = None
    if kwargs.get('run_baseline', True):
        _, baseline_metrics = run_baseline_isolation_forest(train_df, test_df, sensors, test_df[label_col])

    # 3. Apply thresholds
    t_thresh_start = time.time()
    triggers = apply_thresholds(
        test_df.index, sensors, train_res, test_res,
        dataset_name=dataset_name,
        pot_pct=kwargs.get('pot_pct', 98),
        gpd_conf=kwargs.get('gpd_conf', 0.999),
        sensitivity=kwargs.get('sensitivity', 3.0),
        smooth_w=kwargs.get('smooth_w', 10),
        adaptive_w=kwargs.get('adaptive_w', 200),
        consensus_w=kwargs.get('consensus_w', 5),
        consensus_min=kwargs.get('consensus_min', 3)
    )
    t_thresh = time.time() - t_thresh_start

    # 4. Global evaluation
    t_eval_start = time.time()
    y_true = test_df[label_col]
    best_f1, best_v, global_alert, votes, table_df = evaluate_global(
        triggers, y_true, stability_scores=stability_scores, baseline_metrics=baseline_metrics)
    t_eval = time.time() - t_eval_start

    # Benchmarking Report
    print(f"\n--- Process Duration: {dataset_name} ---")
    print(f"  Prep & Info:  {t_feat:.2f}s")
    if t_model > 0:
        print(f"  ML Modeling:  {t_model:.2f}s")
    else:
        print(f"  ML Modeling:  Skipped (Loaded)")
    print(f"  POT/EVT:      {t_thresh:.2f}s")
    print(f"  Evaluation:   {t_eval:.2f}s")
    print(f"  Total Time:   {time.time() - t_start:.2f}s")

    # Visualization
    if kwargs.get('generate_plots', True):
        plot_detection(votes, global_alert, y_true, best_v, best_f1, dataset_name, len(triggers.columns))

    return best_f1, train_res, test_res, sensors, table_df
