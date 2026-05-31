import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLinearClassifier(nn.Module):
    def __init__(self, n_classes=3, dropout=0.2, embedding_dim=768):
        super().__init__()
        self.n_classes = n_classes
        self.norm = nn.LayerNorm(embedding_dim)
        self.attn = nn.Linear(embedding_dim, 1)  # attention scores per residue
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, n_classes)

    def forward(self, x, mask=None):
        # x: (B, L, 768)
        # print(f'x: {x.shape}')
        x = self.norm(x)
        # print(f'x_norm: {x.shape}')

        # attention weights over residues
        scores = self.attn(x).squeeze(-1)  # (B, L)
        # print(f'scores: {scores.shape}')
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=1)  # (B, L)
        # print(f'weights: {weights.shape}')

        # weighted sum -> (B, 768)
        seq_repr = torch.sum(x * weights.unsqueeze(-1), dim=1)

        seq_repr = self.dropout(seq_repr)
        # print(f'seq_repr: {seq_repr.shape}')

        logits = self.fc(seq_repr)
        # print(f'logits: {logits.shape}')
        return logits
