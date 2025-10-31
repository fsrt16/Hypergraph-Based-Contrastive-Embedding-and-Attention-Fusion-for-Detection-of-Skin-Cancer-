# gat_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
# gat_model_enhanced.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

class GATClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, output_dim=64):
        super(GATClassifier, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr)
        return x  # embeddings
    

from GatClassifier import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
class DeepGATConvClassifier2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, heads=4, dropout=0.2, use_projection=True):
        super(DeepGATConvClassifier2, self).__init__()

        # First GAT Layer
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * heads)

        # Intermediate GCN Layer (convolution for spatial smoothing)
        self.conv = GCNConv(hidden_dim * heads, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)

        # Second GAT Layer
        self.gat2 = GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=dropout)
        self.bn3 = torch.nn.BatchNorm1d(output_dim)

        # Projection head for contrastive learning
        self.use_projection = use_projection
        if self.use_projection:
            self.projector = torch.nn.Sequential(
                torch.nn.Linear(output_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64)
            )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index,data.edge_attr

        x = self.gat1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.conv(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.elu(x)

        

        if self.use_projection:
            return self.projector(x)  # For contrastive loss
        print(x.shape)
        return x  # For fine-tuned classification






class TriGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, output_dim=64, num_classes=4):
        super(GATClassifier, self).__init__()
        
        # Inter-Frame Attention via GAT Layers (captures temporal dependencies)
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False)
        
        # Intra-Frame Attention (feature-level attention within each node)
        self.intra_attention = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, 1),
            nn.Softmax(dim=0)
        )

        # Inter-Class Attention Layer (learnable class attention weights)
        self.class_attention = nn.Parameter(torch.Tensor(num_classes, output_dim))
        nn.init.xavier_uniform_(self.class_attention.data)

        # Final classifier (could be replaced with LDA in your framework post embedding)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Inter-Frame Attention
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr)

        # Intra-Frame Attention
        attention_weights = self.intra_attention(x)  # shape: [N, 1]
        x = x * attention_weights  # weighted features

        # Inter-Class Attention
        # Broadcasted cosine similarity with class attention prototypes
        x_norm = F.normalize(x, dim=1)
        class_attn_norm = F.normalize(self.class_attention, dim=1)
        inter_class_scores = torch.matmul(x_norm, class_attn_norm.T)  # shape: [N, num_classes]

        # Combine with original embeddings (or pass inter_class_scores to contrastive loss)
        out = self.classifier(x)

        return out, x, inter_class_scores  # x is the embedding, inter_class_scores for CL loss
