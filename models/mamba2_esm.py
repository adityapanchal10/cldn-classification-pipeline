"""
Mamba-2 inspired classifier on cached ESM2 embeddings.
Pure PyTorch implementation, no external dependencies.
Input: (B, 190, 640) ESM2 embeddings
Output: (B, 3) class logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveStateSpaceBlock(nn.Module):
    """
    Selective state-space layer (Mamba-inspired).
    Pure PyTorch implementation for portability and training stability.
    
    Input: (B, L, D) where D is model dimension
    Output: (B, L, D)
    """
    
    def __init__(self, dim, state_dim=64, expand_factor=2, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        self.expand_factor = expand_factor
        inner_dim = dim * expand_factor
        
        # Projection to hidden dimension
        self.proj_in = nn.Linear(dim, inner_dim * 2)  # hidden + gate
        
        # State-space parameters (learnable)
        self.A_log = nn.Parameter(torch.randn(inner_dim, state_dim) * 0.1)
        self.B = nn.Linear(dim, state_dim)
        self.C = nn.Linear(dim, state_dim)
        self.D = nn.Parameter(torch.ones(inner_dim))
        
        # Gate for selective mechanism
        self.gate_proj = nn.Linear(dim, inner_dim)
        
        # Output
        self.proj_out = nn.Linear(inner_dim, dim)
        
        # Normalization and residual
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def selective_scan(self, u, A, B, C, D):
        """
        Reference selective scan implementation (not currently used in forward).
        Kept for future optimization or bidirectional variants.
        
        Args:
            u: (B, L, dim) input
            A: (dim, state_dim) state matrix
            B: (B, L, state_dim) input projection (selective)
            C: (B, L, state_dim) output projection (selective)
            D: (dim,) feedthrough parameter
            
        Returns:
            output: (B, L, dim)
        """
        B_batch, L, dim = u.shape
        state_dim = A.shape[1]
        device = u.device
        
        # Discretize A: A_d = exp(dt * A) converts continuous to discrete time
        dt = 0.1
        A_d = torch.exp(dt * A)  # (dim, state_dim)
        
        # Initialize state
        h = torch.zeros(B_batch, dim, state_dim, device=device, dtype=u.dtype)
        
        outputs = []
        for t in range(L):
            u_t = u[:, t, :]  # (B, dim)
            B_t = B[:, t, :]  # (B, state_dim)
            C_t = C[:, t, :]  # (B, state_dim)
            
            # Update state: h = A_d * h + B_t @ u_t
            # Reshape for matrix mult: h is (B, dim, state_dim)
            h = torch.einsum('ds,bs->bds', A_d, h.reshape(B_batch, dim, state_dim).squeeze(-1)) + torch.outer(u_t.squeeze(0), B_t.squeeze(0)).unsqueeze(0)
            
            # Simpler recurrence: h_new = A_d * h + B_t * u_t (expanded)
            h = (h * A_d.unsqueeze(0)) + torch.einsum('bs,bd->bds', B_t, u_t)
            
            # Output: y = C_t @ h + D * u_t
            y = torch.einsum('bds,bs->bd', h, C_t) + D.unsqueeze(0) * u_t
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)  # (B, L, dim)
    
    def forward(self, x):
        """
        Selective state-space block forward pass.
        
        Flow: normalize → project to hidden+gate → apply gating → SSM recurrence → project out
        
        Args:
            x: (B, L, dim) input tensor
        
        Returns:
            (B, L, dim) output tensor with residual connection
        """
        residual = x
        x = self.norm(x)  # Pre-normalization
        
        B, L, D = x.shape
        
        # === Expand and Gate ===
        # Project input to hidden dimension (dim → 2*inner_dim: [hidden, gate])
        proj = self.proj_in(x)  # (B, L, 2 * inner_dim)
        inner_dim = self.dim * self.expand_factor
        hidden, gate = proj.chunk(2, dim=-1)  # each (B, L, inner_dim)
        
        # Apply gating: sigmoid controls how much hidden signal passes through
        gate = torch.sigmoid(gate)
        hidden = hidden * gate
        
        # === Generate selective parameters ===
        # B, C control input and output projection per timestep (selective mechanism)
        B_proj = self.B(x)  # (B, L, state_dim) - input selectivity
        C_proj = self.C(x)  # (B, L, state_dim) - output selectivity
        
        # === Selective SSM Recurrence ===
        # Maintain a hidden state that recurrently accumulates information
        h = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)  # (B, state_dim)
        outputs = []
        
        for t in range(L):
            # Aggregate hidden features: compress inner_dim → scalar signal
            hidden_signal = hidden[:, t, :].mean(dim=-1, keepdim=True)  # (B, 1)
            
            # Update state: decay old state + modulate with new input
            h = h * 0.95 + B_proj[:, t, :] * hidden_signal  # (B, state_dim)
            
            # Compute output: pass state through nonlinearity, project back to inner_dim
            state_out = torch.tanh(h)  # (B, state_dim)
            state_contrib = (state_out.mean(dim=-1, keepdim=True) * self.D).expand(B, inner_dim)
            
            # Combine hidden features with state contribution (residual-like)
            y_t = hidden[:, t, :] + state_contrib  # (B, inner_dim)
            outputs.append(y_t)
        
        output = torch.stack(outputs, dim=1)  # (B, L, inner_dim)
        
        # === Project back to original dimension ===
        output = self.proj_out(output)  # (B, L, D)
        output = self.dropout(output)
        
        # Residual connection
        return residual + output


class AttentionPool(nn.Module):
    """
    Attention-based pooling: learns which residues matter most.
    
    For each position in the sequence, compute a learned importance score.
    Compute attention weights via softmax, then weighted average across positions.
    
    Example: if residues 28-81 (extracellular loop) are important for classification,
    attention will learn to assign them higher weights.
    """
    
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Linear(dim, 1)  # Learn importance per position

    def forward(self, x):
        """
        Args:
            x: (B, L, D) sequence of residue embeddings
        
        Returns:
            pooled: (B, D) weighted average across L positions
            attn: (B, L) learned attention weights (inspect for interpretability)
        """
        h = self.norm(x)  # Normalize before scoring
        logits = self.score(h).squeeze(-1)  # (B, L) score per position
        attn = torch.softmax(logits, dim=-1)  # (B, L) normalized weights
        pooled = torch.sum(x * attn.unsqueeze(-1), dim=1)  # (B, D) weighted sum
        return pooled, attn


class Mamba2ESMClassifier(nn.Module):
    """
    Medium-sized Mamba-2-inspired classifier on cached ESM2 embeddings.
    
    Architecture:
    - Input projection: 640 (ESM2) → 256 (internal dim)
    - 6× selective SSM blocks
    - Attention or mean pooling
    - Classification head: 256 → 128 → 3
    
    Args:
        embedding_dim: input dimension (640 for ESM2, 768 for MSA Transformer)
        model_dim: internal model dimension (default 256)
        state_dim: state space dimension (default 64)
        expand_factor: expansion factor for SSM (default 2)
        num_ssm_blocks: number of SSM blocks (default 6)
        dropout: dropout rate (default 0.2)
        num_classes: number of output classes (default 3)
        pooling_mode: 'attention' or 'mean' (default 'attention')
    """
    
    def __init__(
        self,
        embedding_dim=640,
        model_dim=256,
        state_dim=64,
        expand_factor=2,
        num_ssm_blocks=6,
        dropout=0.2,
        num_classes=3,
        pooling_mode='attention',
    ):
        super().__init__()
        
        self.uses_attn = (pooling_mode == 'attention')
        self.pooling_mode = pooling_mode
        self.model_dim = model_dim
        
        # Input projection: ESM2 (640) → model_dim (256)
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.Dropout(dropout),
        )
        
        # SSM blocks
        self.ssm_blocks = nn.ModuleList([
            SelectiveStateSpaceBlock(
                dim=model_dim,
                state_dim=state_dim,
                expand_factor=expand_factor,
                dropout=dropout,
            )
            for _ in range(num_ssm_blocks)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(model_dim)
        
        # Pooling
        if pooling_mode == 'attention':
            self.pool = AttentionPool(model_dim)
            pool_dim = model_dim
        elif pooling_mode == 'mean':
            self.pool = None
            pool_dim = model_dim
        else:
            raise ValueError(f"Unknown pooling mode: {pooling_mode}")
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, model_dim // 2),  # 256 → 128
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, model_dim // 4),  # 128 → 64
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 4, num_classes),  # 64 → 3
        )
    
    def forward(self, x, return_attn=False, return_pooled=False):
        """
        Full Mamba-2 ESM classifier forward pass.
        
        Flow: input_proj → SSM blocks → pooling → classification head
        
        Args:
            x: (B, L, embedding_dim) ESM2 embeddings (640-dim, 190 residues)
            return_attn: if True, also return attention weights (for interpretability)
            return_pooled: if True, also return pooled representation before head
        
        Returns:
            logits: (B, 3) class predictions
            pooled: (B, 256) [optional] aggregated sequence representation
            attn: (B, L) [optional] attention weights per position
        """
        # === Process sequence through SSM blocks ===
        x = self.input_proj(x)  # (B, L, 640) → (B, L, 256)
        
        # Stack 6 SSM blocks for recurrent-style processing
        for block in self.ssm_blocks:
            x = block(x)  # (B, L, 256)
        
        x = self.final_norm(x)  # Final layer normalization
        
        # === Pooling: compress (B, L, 256) → (B, 256) ===
        # Attention pooling: learn which residues matter
        # Mean pooling: uniform average (baseline/ablation)
        if self.pooling_mode == 'attention':
            pooled, attn_weights = self.pool(x)  # (B, 256), (B, L)
        else:  # mean pooling
            pooled = x.mean(dim=1)  # (B, 256)
            attn_weights = None
        
        # === Classification head ===
        class_logits = self.head(pooled)  # (B, 256) → (B, 3)
        
        # === Prepare outputs ===
        outputs = [class_logits]
        
        if return_pooled:
            outputs.append(pooled)
        if return_attn:
            if attn_weights is not None:
                outputs.append(attn_weights)
            else:
                # Mean pooling: return uniform attention as placeholder
                B, L = x.shape[0], x.shape[1]
                outputs.append(torch.ones(B, L, device=x.device) / L)
        
        return tuple(outputs) if len(outputs) > 1 else outputs[0]
