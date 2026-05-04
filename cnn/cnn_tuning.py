import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import optuna
import numpy as np
import warnings

# Suppress minor PyTorch warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. THE MODEL ARCHITECTURE ---
class AudioCNN(nn.Module):
    def __init__(self, in_channels=500, cnn_channels=64, kernel_size=5, hidden_units=128, dropout_rate=0.5, n_layers=1):
        super(AudioCNN, self).__init__()
        
        self.input_norm = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=cnn_channels, kernel_size=kernel_size)
        
        # INCREASED: Max + Avg + Std = cnn_channels * 3
        pooled_dim = cnn_channels * 3 
        
        self.n_layers = n_layers
        self.dropout = nn.Dropout(p=dropout_rate)
        
        if self.n_layers == 0:
            self.fc_out = nn.Linear(pooled_dim, 3)
        elif self.n_layers == 1:
            self.fc1 = nn.Linear(pooled_dim, hidden_units)
            self.fc_out = nn.Linear(hidden_units, 3)
        elif self.n_layers == 2:
            self.fc1 = nn.Linear(pooled_dim, hidden_units)
            self.fc2 = nn.Linear(hidden_units, hidden_units // 2)
            self.fc_out = nn.Linear(hidden_units // 2, 3)

    def forward(self, x):
        x = self.input_norm(x)
        x = F.relu(self.conv1(x))
        
        max_pool = F.adaptive_max_pool1d(x, 1).squeeze(2) 
        avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(2) 
        # ADDED: Standard Deviation Pooling across the time dimension
        std_pool = torch.std(x, dim=2, keepdim=False) 
        
        # Concatenate all three
        x = torch.cat((max_pool, avg_pool, std_pool), dim=1) 
        
        if self.n_layers > 0:
            x = self.dropout(F.relu(self.fc1(x)))
        if self.n_layers == 2:
            x = self.dropout(F.relu(self.fc2(x)))
            
        logits = self.fc_out(x)
        return logits


# --- 2. THE OPTUNA OBJECTIVE (K-FOLD) ---
def objective(trial, dataset):
    # Hyperparameter Search Space
    cnn_channels = trial.suggest_int("cnn_channels", 16, 128, step=16)
    kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    n_layers = trial.suggest_int("n_layers", 0, 2)
    hidden_units = trial.suggest_int("hidden_units", 32, 256, step=32) if n_layers > 0 else 0
    dropout_rate = trial.suggest_float("dropout_rate", 0.4, 0.8) # Increased floor due to overfitting
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True) # Increased ceiling
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"Params: CNN_Ch={cnn_channels}, Layers={n_layers}, Drop={dropout_rate:.2f}, LR={lr:.5f}, WD={weight_decay:.5f}")

    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_val_losses = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        train_sub = Subset(dataset, train_ids)
        val_sub = Subset(dataset, val_ids)
        
        train_loader = DataLoader(train_sub, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=1, shuffle=False)

        # Dynamic Class Weighting (With Clamp Fix)
        all_train_labels = torch.stack([y for _, y in train_sub])
        total_samples = len(train_sub)
        class_sums = torch.clamp(all_train_labels.sum(dim=0), min=1.0)
        raw_pos_weight = (total_samples - class_sums) / class_sums
        pos_weight = torch.clamp(raw_pos_weight, max=5.0) # Prevent explosion
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        model = AudioCNN(cnn_channels=cnn_channels, kernel_size=kernel_size, 
                         hidden_units=hidden_units, dropout_rate=dropout_rate, n_layers=n_layers)
        
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

        # ... [Previous setup: model, optimizer, criterion_train, criterion_val_unweighted] ...
        
        epochs = 100 # Set a high theoretical ceiling
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion_train = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion_val_unweighted = nn.BCEWithLogitsLoss()
        # --- Early Stopping Setup ---
        patience = 7
        epochs_no_improve = 0
        best_fold_val_loss = float('inf')

        # --- Training & Validation Loop per Epoch ---
        for epoch in range(epochs): 
            # 1. Train for one epoch
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion_train(logits, batch_y)
                loss.backward()
                optimizer.step()
            
            scheduler.step()

            # 2. Evaluate immediately
            model.eval()
            current_val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    logits = model(batch_x)
                    loss = criterion_val_unweighted(logits, batch_y) 
                    current_val_loss += loss.item()
            
            current_val_loss /= len(val_loader)

            # 3. Early Stopping Logic
            if current_val_loss < best_fold_val_loss:
                best_fold_val_loss = current_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                # Stop training early to prevent overfitting and save time
                break 

        # Append the BEST loss achieved before overfitting, not the final degraded loss
        fold_val_losses.append(best_fold_val_loss)
        print(f"  Fold {fold + 1}/5 Best Unweighted Val BCE: {best_fold_val_loss:.4f} (Stopped at epoch {epoch+1})")
        # --- Validation Loop ---
        model.eval()
        val_loss_unweighted = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                
                # Calculate the raw, uninflated BCE loss exactly like the KNN script
                loss = criterion_val_unweighted(logits, batch_y) 
                val_loss_unweighted += loss.item()
                
        avg_fold_val_loss = val_loss_unweighted / len(val_loader)
        fold_val_losses.append(avg_fold_val_loss)
        print(f"  Fold {fold + 1}/5 Unweighted Val BCE: {avg_fold_val_loss:.4f}")

    avg_overall_val_loss = np.mean(fold_val_losses)
    print(f"Trial {trial.number} Final Avg Unweighted Val BCE: {avg_overall_val_loss:.4f}")
    
    return avg_overall_val_loss                 


# --- 3. EXECUTION ---
if __name__ == "__main__":
    try:
        full_train_dataset = torch.load("train_dataset.pt")
    except FileNotFoundError:
        print("Dataset not found. Run preprocessing first.")
        exit()

    # Create Optuna Study
    study = optuna.create_study(direction="minimize")
    
    print(f"Starting rigorous K-Fold tuning on {len(full_train_dataset)} training samples...")
    # Increase n_trials if you want a more exhaustive search
    study.optimize(lambda trial: objective(trial, full_train_dataset), n_trials=30)

    print("\n======================================")
    print("BEST HYPERPARAMETERS FOUND:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best Validation Loss: {study.best_value:.4f}")
    print("======================================")