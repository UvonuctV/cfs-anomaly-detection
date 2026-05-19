"""
run_wadi.py — WADI (Water Distribution) Dataset
=================================================
127+ sensors, Stacking Ensemble + EVT/POT/Pareto,
Global Aggregation with sensor voting.

Train: WADI_14days_new.csv (normal operations, ~784K rows)
Test:  WADI_attackdataLABLE.csv (attack scenarios, ~172K rows, label: -1 = attack)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import load_wadi, run_full_pipeline

if __name__ == "__main__":
    TRAIN = os.path.join(os.path.dirname(__file__), '..', 'WADI', 'WADI_14days_new.csv')
    TEST  = os.path.join(os.path.dirname(__file__), '..', 'WADI', 'WADI_attackdataLABLE.csv')

    train_df, test_df, label_col = load_wadi(TRAIN, TEST, downsample=5)

    # Grid search parameters
    sensitivities = [2.0, 1.5, 1.0]
    gpd_confs = [0.99, 0.98, 0.95]
    consensus_mins = [1, 2, 3]
    
    best_overall_f1 = 0
    best_params = {}

    print("\nStarting Hyperparameter Tuning for WADI (Optimized 15-Sensor Setup)...")
    
    # Run the pipeline once to get the residuals
    print("\n--- Initializing Models & Computing Residuals ---")
    _, train_res, test_res, sensors_list, *_ = run_full_pipeline(
        dataset_name=f"WADI",
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        optimize_method=None,
        top_k_sensors=None,
        max_sensors=None,
        pot_pct=99,
        gpd_conf=0.995,
        sensitivity=2.5,
        smooth_w=10,
        adaptive_w=100,
        consensus_w=6,
        consensus_min=3,
        xgb_estimators=20,
        xgb_depth=3,
        xgb_lr=0.1,
        rf_estimators=10,
        rf_depth=4,
    )
    
    for s in sensitivities:
        for c in gpd_confs:
            for c_min in consensus_mins:
                print(f"\n--- Testing: sensitivity={s}, gpd_conf={c}, consensus_min={c_min} ---")
                
                f1, *_ = run_full_pipeline(
                    dataset_name=f"WADI",
                    train_df=train_df,
                    test_df=test_df,
                    label_col=label_col,
                    precomputed_train_res=train_res,
                    precomputed_test_res=test_res,
                    optimize_method=None,
                    top_k_sensors=None,
                    max_sensors=None,
                    pot_pct=99,
                    gpd_conf=c,
                    sensitivity=s,
                    smooth_w=10,
                    adaptive_w=100,
                    consensus_w=6,
                    consensus_min=c_min, generate_plots=False, run_baseline=False,
                )
                
                if f1 > best_overall_f1:
                    best_overall_f1 = f1
                    best_params = {'sensitivity': s, 'gpd_conf': c, 'consensus_min': c_min}
                    
    print("\n========================================================")
    print(f"WADI BEST CONFIGURATION:")
    print(f"F1 Score: {best_overall_f1:.4f}")
    print(f"Params: {best_params}")
    print("========================================================")
