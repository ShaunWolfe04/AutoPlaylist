import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftProtoNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SoftProtoNet, self).__init__()
        # Optional: Project MusiCNN features into a better metric space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Learnable scale and bias for the Sigmoid activation
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        
    def compute_prototypes(self, support_embeddings, support_labels, eps=1e-8):
        """
        support_embeddings: [num_support, embed_dim]
        support_labels: [num_support, num_classes] (Scores from 0 to 1)
        """
        # Transpose labels to [num_classes, num_support] for matrix multiplication
        weights = support_labels.t()
        
        # Weighted sum: [num_classes, num_support] @ [num_support, embed_dim] -> [num_classes, embed_dim]
        weighted_sum = torch.matmul(weights, support_embeddings)
        
        # Sum of weights per class: [num_classes, 1]
        weight_sum_per_class = weights.sum(dim=1, keepdim=True)
        
        # Divide to get the weighted average (with eps to prevent div by zero)
        prototypes = weighted_sum / (weight_sum_per_class + eps)
        return prototypes

    def forward(self, query_embeddings, prototypes):
        """
        query_embeddings: [num_queries, embed_dim]
        prototypes: [num_classes, embed_dim]
        """
        # Calculate squared Euclidean distance
        # Shape: [num_queries, num_classes]
        distances = torch.cdist(query_embeddings, prototypes) ** 2
        
        # Constrain alpha to be strictly positive to prevent gradient collapse
        safe_alpha = F.softplus(self.alpha)
        
        # Shift distances and apply Sigmoid
        logits = -safe_alpha * distances + self.beta
        probabilities = torch.sigmoid(logits)
        
        return probabilities