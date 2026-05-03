import train
import optuna


study = optuna.create_study(
    direction="minimize",
    study_name="Soft Prototypical Network Param Optimization",
    load_if_exists = True,
    storage="sqlite:///playlist_model.db"
    )

study.optimize(train.objective, n_trials=None, timeout=600, n_jobs = 4) #timeout is in seconds

# Print the best results
print("\n--- OPTIMIZATION FINISHED ---")
print("Best Trial:")
trial = study.best_trial

print(f"  Value (Val Loss): {trial.value}")
print("  Best Hyperparameters:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")