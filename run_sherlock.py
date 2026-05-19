import os
import sys

# Ensure pipeline is accessible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pipeline import load_sherlock, run_full_pipeline

if __name__ == "__main__":
    TRAIN = os.path.join(os.path.dirname(__file__), '..', '01-Basic', '01-Basic', 'train_flat.csv')
    TEST  = os.path.join(os.path.dirname(__file__), '..', '01-Basic', '01-Basic', 'test_flat.csv')

    # 1. Load Data
    # 472 columns is massive. No downsampling for maximum fidelity.
    train_df, test_df, label_col = load_sherlock(TRAIN, TEST, downsample=1)

    # 2. Execute Full ML Pipeline
    f1, *_ = run_full_pipeline(
        dataset_name="Sherlock 01_Basic",
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        optimize_method="rf_importance",
        top_k_sensors=50,
        pot_pct=98,           # Standard POT percentile
        gpd_conf=0.999,       # Standard GPD Confidence
        sensitivity=2.0,      # Moderate sensitivity for a new dataset
        smooth_w=5,
        adaptive_w=100,
        consensus_w=5,
        consensus_min=3,      # At least 3 sensors must agree
        xgb_estimators=30,    # Reduced estimators for speed given 472 sensors
        xgb_depth=3,
        xgb_lr=0.1,
        rf_estimators=20,     # Reduced estimators for speed
        rf_depth=4,
    )
    
    print(f"\nSherlock Final F1: {f1:.4f}")
