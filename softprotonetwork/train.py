import torch.optim as optim
from SoftProtoNet import SoftProtoNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from EpisodeGenerator import generate_episode

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
input_dim = 400 #TODO Change this around. for now, itll be 1000 for max and mean pooling
model = SoftProtoNet(input_dim=input_dim, hidden_dim=512, output_dim=256)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# Load training and labels all into memory, which should only take like a gigabyte ish
# train_embeddings should just be [num songs, embedding dimension]
# train_labels should be [num songs, num playlists]
train_embeddings = torch.from_numpy(np.load("../all_embeddings.npy")).float()
print(f"Train embeddings shape: {train_embeddings.shape}")
train_labels = torch.from_numpy(np.load("../all_labels.npy")).float()

#TODO properly make a test and train set

#TODO make the embeddings space size modular

# Training Loop
num_episodes = 5000
model.train()

for episode in range(num_episodes):
    optimizer.zero_grad()
    
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
    loss = criterion(predictions, Q_lab)
    #print(f"Calculated Loss for Episode {episode + 1}")
    #print(f"Computing Backwards step for Episode {episode + 1}")
    loss.backward()
    #print(f"Finished backwards step for Epsiode {episode + 1}")

    enc_grad_norm = get_grad_norm(model.encoder.parameters())
    metric_grad_norm = get_grad_norm([model.alpha, model.beta])

    optimizer.step()
    
    if episode % 100 == 0:
        print(f"Episode {episode} | Loss: {loss.item():.4f} | Alpha: {F.softplus(model.alpha).item():.4f} | Beta: {model.beta.item():.4f} | Enc Grad Norm: {enc_grad_norm:.4f} | AlphaBeta Grad Norm: {metric_grad_norm:.4f}")
        