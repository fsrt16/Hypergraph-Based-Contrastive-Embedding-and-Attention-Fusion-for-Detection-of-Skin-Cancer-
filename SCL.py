import torch
import torch.nn.functional as F  # Remove this if not using F

class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5, eps=1e-8):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, embeddings, labels):
        # Normalize embeddings (L2)
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=self.eps)

        # Compute cosine similarity matrix [N, N]
        sim_matrix = torch.matmul(embeddings, embeddings.T)  # cosine sim because embeddings are normalized

        # Optional debugging
        if torch.isnan(embeddings).any():
            print("NaNs in embeddings")
        if torch.isnan(sim_matrix).any():
            print("NaNs in similarity matrix")

        # Rescale similarities with temperature
        sim_matrix = sim_matrix / self.temperature
        sim_matrix = torch.clamp(sim_matrix, min=-20, max=20)  # stability clamp
        exp_sim = torch.exp(sim_matrix)

        #print(f"Similarity Matrix Shape: {sim_matrix.shape}")

        # Get labels into shape for pairwise comparison
        labels = labels.view(-1, 1)
        pos_mask = labels == labels.T  # [N, N]

        # Remove self-comparisons
        self_mask = torch.eye(sim_matrix.size(0), device=embeddings.device).bool()
        pos_mask = pos_mask.masked_fill(self_mask, False)  # Don't compare i to itself

        # Denominator: sum over all exp(s_ij) except self
        denom = exp_sim.masked_fill(self_mask, 0).sum(dim=1, keepdim=True) + self.eps

        # Numerator: sum of exp(s_ij) over positive pairs
        numerator = exp_sim * pos_mask
        numerator_sum = numerator.sum(dim=1, keepdim=True)

        # Compute the supervised contrastive loss
        loss = -torch.log(numerator_sum / denom)
        loss = loss[pos_mask.sum(dim=1) > 0]  # Only consider nodes with at least one positive
        return loss.mean()