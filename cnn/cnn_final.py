import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import csv

# --- 1. PASTE OPTUNA RESULTS HERE ---
# Replace these with the exact outputs from your cnn_tuning.py terminal
BEST_PARAMS = {
    "cnn_channels": 80,
    "kernel_size": 7,
    "n_layers": 0,
    "hidden_units": 128,
    "dropout_rate": 0.7481197870110796,
    "learning_rate": 3.09048889773453e-05,
    "weight_decay": 0.0014370139270072223,
    "optimizer": "AdamW"
}

EPOCHS_TO_TRAIN = 40 # Adjust this to roughly match where your best trial triggered Early Stopping

# --- 2. THE MODEL ARCHITECTURE ---
class AudioCNN(nn.Module):
    def __init__(self, in_channels=500, cnn_channels=64, kernel_size=5, hidden_units=128, dropout_rate=0.5, n_layers=1):
        super(AudioCNN, self).__init__()
        
        self.input_norm = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=cnn_channels, kernel_size=kernel_size)
        
        # Max + Avg + Std pooling
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
        std_pool = torch.std(x, dim=2, keepdim=False) 
        
        x = torch.cat((max_pool, avg_pool, std_pool), dim=1) 
        
        if self.n_layers > 0:
            x = self.dropout(F.relu(self.fc1(x)))
        if self.n_layers == 2:
            x = self.dropout(F.relu(self.fc2(x)))
            
        logits = self.fc_out(x)
        return logits

# --- 3. CSV EXPORT FUNCTION ---
def evaluate_and_save(model, dataloader, split_name, csv_writer):
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            
            # Convert logits to 0.0 - 1.0 probabilities
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            true_labels = batch_y.squeeze(0).cpu().numpy()
            
            row = [split_name] + list(true_labels) + list(probabilities)
            csv_writer.writerow(row)

# --- 4. EXECUTION ---
def main():
    try:
        train_dataset = torch.load("train_dataset.pt")
        test_dataset = torch.load("test_dataset.pt")
    except FileNotFoundError:
        print("Error: Dataset files not found.")
        return

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    eval_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Dynamic Class Weighting
    all_train_labels = torch.stack([y for _, y in train_dataset])
    total_samples = len(train_dataset)
    class_sums = torch.clamp(all_train_labels.sum(dim=0), min=1.0)
    raw_pos_weight = (total_samples - class_sums) / class_sums
    pos_weight = torch.clamp(raw_pos_weight, max=5.0)

    # Initialize model with Best Params
    model = AudioCNN(
        cnn_channels=BEST_PARAMS["cnn_channels"],
        kernel_size=BEST_PARAMS["kernel_size"],
        hidden_units=BEST_PARAMS["hidden_units"],
        dropout_rate=BEST_PARAMS["dropout_rate"],
        n_layers=BEST_PARAMS["n_layers"]
    )
    
    optimizer = getattr(optim, BEST_PARAMS["optimizer"])(
        model.parameters(), 
        lr=BEST_PARAMS["learning_rate"], 
        weight_decay=BEST_PARAMS["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_TO_TRAIN)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Training Final Model for {EPOCHS_TO_TRAIN} epochs...")
    for epoch in range(EPOCHS_TO_TRAIN):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS_TO_TRAIN} | Train Loss: {total_loss / len(train_loader):.4f}")

    # Generate CSV
    csv_filename = "cnn_predictions.csv"
    print(f"\nTraining complete. Saving predictions to {csv_filename}...")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Split', 'True_Study', 'True_Drive', 'True_Workout', 'Pred_Study', 'Pred_Drive', 'Pred_Workout'])
        
        evaluate_and_save(model, eval_train_loader, "Train", writer)
        evaluate_and_save(model, test_loader, "Test", writer)
        
    print("Done.")

if __name__ == "__main__":
    main()