import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.4):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class SingleSequenceAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn = ResidualMLPBlock(dim, hidden_dim=dim * 2, dropout=dropout)

    def forward(self, x):
        x_norm = self.norm(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        x = self.ffn(x)
        return x


class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        h = self.norm(x)
        logits = self.score(h).squeeze(-1)  # (B, R)
        attn = torch.softmax(logits, dim=-1)  # (B, R)
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)  # (B, D)
        return pooled, attn


class TransformerMLPClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim=768,  # ESM embedding dim — fixed, 768 for MSA Transformer, 640 for ESM2
        proj_dim=128,
        num_classes=3,
        num_heads=4,
        num_attention_blocks=1,
        dropout=0.4,
        max_seq_len=220,
        embedding_noise_std=0.02
    ):
        super().__init__()

        self.uses_attn = True

        # Stage 1: Project ESM embeddings down to a manageable size
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, proj_dim),  # 768/640 → 128
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
        )

        # Stage 2: Positional embeddings (now 128-dim, not 768/640-dim)
        self.pos_emb = nn.Embedding(max_seq_len, proj_dim)

        self.emb_norm_before = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)

        # Stage 3: Single attention block (was 2)
        self.attention_blocks = nn.ModuleList(
            [
                SingleSequenceAttentionBlock(
                    dim=proj_dim, num_heads=num_heads, dropout=dropout
                )
                for _ in range(num_attention_blocks)
            ]
        )

        self.emb_norm_after = nn.LayerNorm(proj_dim)

        # Stage 4: Attention pooling
        self.residue_pool = AttentionPool(proj_dim)

        # Stage 5: Lightweight classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim // 2),  # 128 → 64
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, proj_dim // 4),  # 64 → 32
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 4, num_classes),  # 32 → 3
        )

    def forward(self, x, return_attn=False, return_pooled=False):
        """
        x: (B, R, D) — batch, residues, ESM embedding dim (768/640)
        """
        B, R, D = x.shape
        device = x.device

        # Project 768/640 → 128
        x = self.input_proj(x)  # (B, R, 128)

        # Add Gaussian noise during training
        if self.training:
            x = x + torch.randn_like(x) * self.embedding_noise_std

        # Add positional embeddings
        res_ids = torch.arange(R, device=device)
        pos_emb = self.pos_emb(res_ids)[None, :, :]  # (1, R, 128)
        x = x + pos_emb

        # Pre-norm + dropout
        x = self.emb_norm_before(x)
        x = self.dropout(x)

        # Attention block(s)
        for block in self.attention_blocks:
            x = block(x)

        x = self.emb_norm_after(x)  # (B, R, 128)

        # Attention pooling → single vector per sequence
        pooled, residue_attn = self.residue_pool(x)  # (B, 128)

        # Classify
        class_logits = self.head(pooled)  # (B, 3)

        outputs = [class_logits]

        if return_pooled:
            outputs.append(pooled)
        if return_attn:
            outputs.append(residue_attn)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]
