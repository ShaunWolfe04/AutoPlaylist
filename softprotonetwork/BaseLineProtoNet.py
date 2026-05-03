import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineSoftProtoNet(nn.Module):
    def __init__(self):
        super(BaselineSoftProtoNet, self).__init__()
        
        # nn.Identity() simply returns whatever input you give it.
        # This allows us to keep `model.encoder(embeddings)` in the training loop
        # without it actually altering the MusiCNN features!
        self.encoder = nn.Identity()
        
        # We STILL learn the scale and bias to map raw MusiCNN distances to probabilities
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        
    def compute_prototypes(self, support_embeddings, support_labels, eps=1e-8):
        """
        Calculates prototypes exactly as before, but now operating purely
        on the raw, unprojected MusiCNN features.
        """
        weights = support_labels.t()
        weighted_sum = torch.matmul(weights, support_embeddings)
        weight_sum_per_class = weights.sum(dim=1, keepdim=True)
        prototypes = weighted_sum / (weight_sum_per_class + eps)
        return prototypes

    def forward(self, query_embeddings, prototypes):
        """
        Calculates distances and maps them to probabilities.
        """
        # Distance calculation is identical
        distances = torch.cdist(query_embeddings, prototypes) ** 2

        embed_dim = query_embeddings.shape[-1]
        distances = distances / embed_dim
        
        # Alpha and Beta do all the heavy lifting here
        safe_alpha = F.softplus(self.alpha)
        logits = -safe_alpha * distances + self.beta
        #probabilities = torch.sigmoid(logits)
        
        return logits