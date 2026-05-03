import torch.optim as optim
from SoftProtoNet import SoftProtoNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from EpisodeGenerator import generate_episode

#TODO mess with these as needed
INPUT_DIM = 400
TRAIN_SPLIT = 0.5
TEST_SPLIT = 0.2
VAL_SPLIT = 0.3
MAX_EPISODES = 100000
VAL_EVERY = 100
LR_DECAY_EVERY = 1500
BATCH_EPISODE_COUNT = 4

assert TRAIN_SPLIT + TEST_SPLIT + VAL_SPLIT == 1.0

def get_grad_norm(parameters):
    """Calculates the Global L2 Norm of gradients."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            # Calculate the L2 norm of the gradient for this parameter
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# Initialization
input_dim = INPUT_DIM #TODO Change this around. for now, itll be 1000 for max and mean pooling
model = SoftProtoNet(input_dim=input_dim, hidden_dim=512, output_dim=256)
#optimizer = optim.Adam(model.parameters(), lr=1e-2)
optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-6},
    {'params': [model.alpha, model.beta], 'lr': 0.025}
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_EVERY, gamma = 0.5) #learning rate halving every 2000 episodes as described by the original paper #TODO FACT CHECK
criterion = nn.BCELoss()


# Load training and labels all into memory, which should only take like a gigabyte ish
# train_embeddings should just be [num songs, embedding dimension]
# train_labels should be [num songs, num playlists]
embeddings = torch.from_numpy(np.load("../all_embeddings.npy")).float()
labels = torch.from_numpy(np.load("../all_labels.npy")).float()

# Split into train test val
num_songs = embeddings.shape[0]
indics = np.random.permutation(num_songs)
train_split = int(TRAIN_SPLIT * num_songs)
val_split = int((TRAIN_SPLIT + VAL_SPLIT) * num_songs)

train_idx = indics[:train_split]
val_idx = indics[train_split:val_split]
test_idx = indics[val_split:]

train_embeddings = embeddings[train_idx]
train_labels = labels[train_idx]
val_embeddings = embeddings[val_idx]
val_labels = labels[val_idx]
test_embeddings = embeddings[test_idx]
test_labels = embeddings[test_idx]
print(f"Songs -> Train: {train_embeddings.shape[0]} | Val: {val_embeddings.shape[0]} | Test: {test_embeddings.shape[0]}")




#TODO properly impl train/test/val, lr decay, early stopping

#TODO impl


# Training Loop
num_episode_batches = int(MAX_EPISODES / BATCH_EPISODE_COUNT)
model.train()

for episode_batch in range(num_episode_batches):
    optimizer.zero_grad()

    for episode in range(BATCH_EPISODE_COUNT):
        
    
        # Get episodic data
        #print(f"Generating Episode {episode + 1}")
        S_emb, S_lab, Q_emb, Q_lab, _ = generate_episode(train_embeddings, train_labels, num_classes_per_episode=5)
        #print(f"Generated Episode {episode + 1}")
        
        # Project embeddings into the learned metric space
        #print(f"Encoding Embeddings of Episode {episode + 1}")
        S_encoded = model.encoder(S_emb)
        Q_encoded = model.encoder(Q_emb)
        #print(f"Encoded Embeddings of Episode {episode + 1}")
        
        # Calculate prototypes and predictions
        #print(f"Calculating Prototypes for Episode {episode + 1}")
        prototypes = model.compute_prototypes(S_encoded, S_lab)
        #print(f"Finished Calculating Prototypes for Episode {episode + 1}")
        #print(f"Calculting Predictions for Episode {episode + 1}")
        predictions = model(Q_encoded, prototypes)
        #print(f"Calculated Predictions for Episode {episode + 1}")
        
        # Calculate loss and backpropagate
        #print(f"Calculting Loss for Episode {episode + 1}")
        loss = criterion(predictions, Q_lab) / BATCH_EPISODE_COUNT



        #print(f"Calculated Loss for Episode {episode + 1}")
        #print(f"Computing Backwards step for Episode {episode + 1}")
        loss.backward()
        #print(f"Finished backwards step for Epsiode {episode + 1}")

    optimizer.step()
    scheduler.step()
    
    if episode_batch % int(VAL_EVERY / BATCH_EPISODE_COUNT) == 0:
        enc_grad_norm = get_grad_norm(model.encoder.parameters())
        metric_grad_norm = get_grad_norm([model.alpha, model.beta])

        # run an episode on the validation set
        model.eval()
        with torch.no_grad():
            val_S_emb, val_S_lab, val_Q_emb, val_Q_lab, _ = generate_episode(
                val_embeddings, val_labels, num_classes_per_episode=5
            )
            
            val_S_enc = model.encoder(val_S_emb)
            val_Q_enc = model.encoder(val_Q_emb)
            
            val_protos = model.compute_prototypes(val_S_enc, val_S_lab)
            val_preds = model(val_Q_enc, val_protos)
            val_loss = criterion(val_preds, val_Q_lab).item()
            
            # current_lr = scheduler.get_last_lr()[0]
            
            #TODO impl this a bit later
            # Check Early Stopping Criteria
            #if val_loss < best_val_loss:
            #    best_val_loss = val_loss
            #    epochs_no_improve = 0
            #    # Save the new best weights
            #    best_model_weights = copy.deepcopy(model.state_dict())
            #else:
            #    epochs_no_improve += 1
            #    print(f"-> No improvement ({epochs_no_improve}/{patience})")
            #    
            #if epochs_no_improve >= patience:
            #    print("Early stopping triggered! Restoring best weights.")
            #    model.load_state_dict(best_model_weights)
            #    break # Exit the training loop early
                
        model.train()

        print(f"Episode {episode_batch * BATCH_EPISODE_COUNT} | Train Loss: {loss.item() * BATCH_EPISODE_COUNT:.4f} | Validation Loss: {val_loss:.4f} | Alpha: {F.softplus(model.alpha).item():.4f} | Beta: {model.beta.item():.4f} | Enc Grad Norm: {enc_grad_norm:.4f} | AlphaBeta Grad Norm: {metric_grad_norm:.4f}")
        