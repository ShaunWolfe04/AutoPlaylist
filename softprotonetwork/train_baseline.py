import torch.optim as optim
from BaseLineProtoNet import BaseLineProtoNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from EpisodeGenerator import generate_episode

# Initialization
input_dim = 1000 #TODO Change this around. for now, itll be 1000 for max and mean pooling
model = BaseLineProtoNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

# Training Loop
num_episodes = 5000
model.train()

for episode in range(num_episodes):
    optimizer.zero_grad()
    
    # Get episodic data
    S_emb, S_lab, Q_emb, Q_lab, _ = generate_episode(train_embeddings, train_labels, num_classes_per_episode=5)
    
    # Project embeddings into the learned metric space
    S_encoded = model.encoder(S_emb)
    Q_encoded = model.encoder(Q_emb)
    
    # Calculate prototypes and predictions
    prototypes = model.compute_prototypes(S_encoded, S_lab)
    predictions = model(Q_encoded, prototypes)
    
    # Calculate loss and backpropagate
    loss = criterion(predictions, Q_lab)
    loss.backward()
    optimizer.step()
    
    if episode % 500 == 0:
        print(f"Episode {episode} | Loss: {loss.item():.4f} | Alpha: {F.softplus(model.alpha).item():.4f} | Beta: {model.beta.item():.4f}")