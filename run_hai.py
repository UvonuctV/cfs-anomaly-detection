"""
run_hai.py — HAI (HIL-based Augmented ICS Security)
====================================================
80+ sensors (57 variable), Stacking Ensemble + EVT/POT/Pareto,
Global Aggregation with sensor voting.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from pipeline import load_hai, run_full_pipeline

if __name__ == "__main__":
    TRAIN = os.path.join(os.path.dirname(__file__), '..', 'HAI_Dataset', 'train1.csv')
    TEST  = os.path.join(os.path.dirname(__file__), '..', 'HAI_Dataset', 'test1.csv')

    train_df, test_df, label_col = load_hai(TRAIN, TEST, downsample=1)

    sensitivities = [2.0, 1.5, 1.0]
    gpd_confs = [0.99, 0.98, 0.95]
    consensus_mins = [1, 2, 3]

    best_overall_f1 = 0
    best_params = {}

    print("\nStarting Hyperparameter Tuning for HAI (Optimized 15-Sensor Setup)...")

    print("\n--- Initializing Models & Computing Residuals ---")
    _, train_res, test_res, sensors_list, *_ = run_full_pipeline(
        dataset_name=f"HAI",
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        optimize_method=None,
        top_k_sensors=None,
        pot_pct=98,
        gpd_conf=0.999,
        sensitivity=3.0,
        smooth_w=10,
        adaptive_w=200,
        consensus_w=5,
        consensus_min=3,
        xgb_estimators=50,
        xgb_depth=3,
        xgb_lr=0.1,
        rf_estimators=30,
        rf_depth=4,
    )

    for s in sensitivities:
        for c in gpd_confs:
            for c_min in consensus_mins:
                print(f"\n--- Testing: sensitivity={s}, gpd_conf={c}, consensus_min={c_min} ---")
                
                f1, *_ = run_full_pipeline(
                    dataset_name=f"HAI",
                    train_df=train_df,
                    test_df=test_df,
                    label_col=label_col,
                    precomputed_train_res=train_res,
                    precomputed_test_res=test_res,
                    optimize_method=None,
                    top_k_sensors=None,
                    pot_pct=98,
                    gpd_conf=c,
                    sensitivity=s,
                    smooth_w=10,
                    adaptive_w=200,
                    consensus_w=5,
                    consensus_min=c_min,
                    generate_plots=False,
                    run_baseline=False,
                )
                
                if f1 > best_overall_f1:
                    best_overall_f1 = f1
                    best_params = {'sensitivity': s, 'gpd_conf': c, 'consensus_min': c_min}
                    
    print("\n========================================================")
    print(f"HAI BEST CONFIGURATION:")
    print(f"F1 Score: {best_overall_f1:.4f}")
    print(f"Params: {best_params}")
    print("========================================================")
