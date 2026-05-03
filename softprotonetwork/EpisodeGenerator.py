import numpy as np
import torch

def generate_episode(all_embeddings, all_labels, num_classes_per_episode, anchors_per_class=2, num_fillers=15, num_queries=15):
    """
    Conceptual generator for a single training episode.
    all_embeddings: Tensor of all training MusiCNN features
    all_labels: Tensor of all Likert soft labels [total_songs, total_playlists]
    """
    total_playlists = all_labels.shape[1]
    
    # 1. Select random playlists for this episode
    # For now, we are bypassing this and using all classes since we only have 3
    #episode_classes = np.random.choice(total_playlists, num_classes_per_episode, replace=False)
    episode_classes = np.arange(total_playlists)

    support_indices = set()
    
    # 2. Sample Anchors (Ensure stability)
    for c in episode_classes:
        # Find indices where the score for playlist 'c' is >= 0.75
        strong_matches = torch.where(all_labels[:, c] >= 0.75)[0].tolist()
        if len(strong_matches) > 0:
            # Randomly pick anchors
            sampled_anchors = np.random.choice(strong_matches, min(anchors_per_class, len(strong_matches)), replace=False)
            support_indices.update(sampled_anchors)
            
    # 3. Sample Fillers (The overlapping nuance)
    available_fillers = list(set(range(len(all_labels))) - support_indices)
    fillers = np.random.choice(available_fillers, num_fillers, replace=False)
    support_indices.update(fillers)
    
    support_indices = list(support_indices)
    
    # 4. Sample Queries (Must be disjoint from support)
    available_queries = list(set(range(len(all_labels))) - set(support_indices))
    query_indices = np.random.choice(available_queries, min(num_queries, len(available_queries)), replace=False)
    
    # Return the split data sliced to only the classes in this episode
    S_emb, S_lab = all_embeddings[support_indices], all_labels[support_indices][:, episode_classes]
    Q_emb, Q_lab = all_embeddings[query_indices], all_labels[query_indices][:, episode_classes]
    
    return S_emb, S_lab, Q_emb, Q_lab, episode_classes

