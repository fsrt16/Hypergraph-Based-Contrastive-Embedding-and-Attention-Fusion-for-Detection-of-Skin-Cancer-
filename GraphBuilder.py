# graph_builder.py
import torch
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def build_graph(X_feats, Y_labels, threshold=0.95, eval =False):
    # Get the number of samples (nodes)
    num_samples = X_feats.shape[0]
    if eval:
        # with self edges and weoghts of 1.0
        return Data(x=torch.tensor(X_feats, dtype=torch.float), 
                edge_index=torch.arange(num_samples).unsqueeze(0).repeat(2, 1),
                edge_attr=torch.ones(num_samples, dtype=torch.float),
                     y=torch.tensor(Y_labels, dtype=torch.long))
    # Calculate the cosine similarity matrix
    sim_matrix = cosine_similarity(X_feats)

    # Initialize lists to hold the edge indices and weights
    edge_index, edge_weight = [], []
    
    for i in range(num_samples):
        for j in range(num_samples):
            if i == j:
                continue
            sim = sim_matrix[i][j]
            if Y_labels[i] == Y_labels[j] and sim > threshold:
                edge_index.append([i, j])  # Same class, positive similarity
                edge_weight.append(sim)
            elif Y_labels[i] != Y_labels[j] and sim > threshold:
                edge_index.append([i, j])  # Different class, negative similarity
                edge_weight.append(-sim)

    # Convert edge_index to tensor and transpose to match expected format (2, num_edges)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    # Convert node features and labels to tensors
    X_tensor = torch.tensor(X_feats, dtype=torch.float)
    Y_tensor = torch.tensor(Y_labels, dtype=torch.long)

    # Return as PyTorch Geometric Data object
    return Data(x=X_tensor, edge_index=edge_index, edge_attr=edge_weight, y=Y_tensor)
