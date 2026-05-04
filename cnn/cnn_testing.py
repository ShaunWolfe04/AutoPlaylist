import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import csv

# --- 1. THE MODEL ARCHITECTURE (WITH FIXES) ---
class AudioCNN(nn.Module):
    def __init__(self, in_channels=500, cnn_channels=64, kernel_size=5, hidden_units=128, dropout_rate=0.5, n_layers=1):
        super(AudioCNN, self).__init__()
        
        # FIX: Normalize the raw input to prevent exploding gradients
        self.input_norm = nn.BatchNorm1d(in_channels)
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=cnn_channels, kernel_size=kernel_size)
        pooled_dim = cnn_channels * 2
        
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
        # Apply normalization before convolution
        x = self.input_norm(x)
        x = F.relu(self.conv1(x))
        
        max_pool = F.adaptive_max_pool1d(x, 1).squeeze(2) 
        avg_pool = F.adaptive_avg_pool1d(x, 1).squeeze(2) 
        
        x = torch.cat((max_pool, avg_pool), dim=1) 
        
        if self.n_layers > 0:
            x = self.dropout(F.relu(self.fc1(x)))
        if self.n_layers == 2:
            x = self.dropout(F.relu(self.fc2(x)))
            
        logits = self.fc_out(x)
        return logits

# --- 2. CSV EXPORT FUNCTION ---
def evaluate_and_save(model, dataloader, split_name, csv_writer):
    """Evaluates the model and writes true labels and predictions to the CSV."""
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            logits = model(batch_x)
            
            # Convert raw logits to probabilities (0.0 to 1.0) for the CSV
            probabilities = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            true_labels = batch_y.squeeze(0).cpu().numpy()
            
            # Format: [Split, True_S, True_D, True_W, Pred_S, Pred_D, Pred_W]
            row = [split_name] + list(true_labels) + list(probabilities)
            csv_writer.writerow(row)

# --- 3. EXECUTION ---
def main():
    try:
        train_dataset = torch.load("train_dataset.pt")
        test_dataset = torch.load("test_dataset.pt")
    except FileNotFoundError:
        print("Error: Dataset files not found. Run preprocessing first.")
        return

    # DataLoaders (Batch size 1 for variable length T)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    # Unshuffled train_loader for clean evaluation output
    eval_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Dynamic Class Weighting (With Clamp Fix)
    all_train_labels = torch.stack([y for _, y in train_dataset])
    total_samples = len(train_dataset)
    class_sums = torch.clamp(all_train_labels.sum(dim=0), min=1.0)
    raw_pos_weight = (total_samples - class_sums) / class_sums
    
    # FIX: Clamp the max weight to 5.0 to prevent severe oscillation
    pos_weight = torch.clamp(raw_pos_weight, max=5.0)
    print(f"Calculated pos_weight: {pos_weight.numpy()}")

    # Initialize model, loss, and optimizer with static parameters
    model = AudioCNN(cnn_channels=64, kernel_size=7, n_layers=0, dropout_rate=0.51)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=5.9e-5, weight_decay=0.013)

    # Training Loop
    # Training Loop
    epochs = 15
    print("\nStarting Training...")
    for epoch in range(epochs):
        
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval() 
        total_val_loss = 0.0
        with torch.no_grad(): 
            for batch_x, batch_y in test_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(test_loader)

        # Print both metrics
        print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    # Evaluation and CSV Writing
    csv_filename = "model_predictions.csv"
    print(f"\nTraining complete. Saving predictions to {csv_filename}...")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write Header
        writer.writerow(['Split', 'True_Study', 'True_Drive', 'True_Workout', 'Pred_Study', 'Pred_Drive', 'Pred_Workout'])
        
        # Write Data
        evaluate_and_save(model, eval_train_loader, "Train", writer)
        evaluate_and_save(model, test_loader, "Test", writer)
        
    print("Done.")

if __name__ == "__main__":
    main()