"""
run_batadal.py — BATADAL Water Distribution Network
====================================================
43 sensors, Stacking Ensemble + EVT/POT/Pareto,
Global Aggregation with sensor voting.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import load_batadal, run_full_pipeline

if __name__ == "__main__":
    TRAIN = os.path.join(os.path.dirname(__file__), '..', 'BATADAL', 'BATADAL_dataset03.csv')
    TEST  = os.path.join(os.path.dirname(__file__), '..', 'BATADAL', 'BATADAL_dataset04.csv')

    train_df, test_df, label_col = load_batadal(TRAIN, TEST)

    f1, *_ = run_full_pipeline(
        dataset_name="BATADAL",
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        optimize_method=None,
        top_k_sensors=None,
        pot_pct=98,
        gpd_conf=0.999,
        sensitivity=3.0,
        smooth_w=10,
        adaptive_w=300,
        consensus_w=5,
        consensus_min=3,
        xgb_estimators=100,
        xgb_depth=4,
        xgb_lr=0.05,
        rf_estimators=50,
        rf_depth=5,
    )
    print(f"\nBATADAL Final F1: {f1:.4f}")
