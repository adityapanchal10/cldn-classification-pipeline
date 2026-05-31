import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNClassifier(nn.Module):
    def __init__(
        self,
        n_classes=3,
        embedding_dim=768,
        n_filters=100,
        filter_sizes=[3, 4, 5],
        dropout=0.1,
    ):
        super().__init__()

        # Normalization layer for input embeddings
        self.norm = nn.LayerNorm(embedding_dim)

        # Define multiple convolutional layers with different filter sizes
        # Each filter looks at 'fs' words at a time across the full embedding width
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=n_filters,
                    kernel_size=(fs, embedding_dim),
                )
                for fs in filter_sizes
            ]
        )

        # Final fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * n_filters, n_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [Batch, Length, 768] - Input embeddings
            mask: [Batch, Length] - Boolean tensor (True for data, False for padding)
        """

        # 1. Normalize and add channel dimension
        # Result: [Batch, 1, Length, 768]
        x = self.norm(x).unsqueeze(1)

        pooled_outputs = []

        for conv in self.convs:
            # 2. Apply Convolution and ReLU
            # conved shape: [Batch, n_filters, L_out, 1]
            conved = F.relu(conv(x)).squeeze(3)

            # 3. Apply Masking Logic
            if mask is not None:
                # Get the filter size (height) of the current convolution
                fs = conv.kernel_size[0]

                # Slicing the mask:
                # Because the convolution reduces the sequence length by (fs - 1),
                # we align the mask by starting from the end of the first window.
                # output_mask shape: [Batch, L_out]
                output_mask = mask[:, fs - 1 :]

                # Expand mask to [Batch, 1, L_out] to match the 'conved' tensor
                output_mask = output_mask.unsqueeze(1)

                # Fill padding regions with a very small number (-1e9)
                # This ensures Max Pooling ignores these positions.
                conved = conved.masked_fill(~output_mask, -1e9)

            # 4. Global Max Pooling
            # Picks the single most important feature per filter
            # Result: [Batch, n_filters]
            pooled = F.max_pool1d(conved, conved.shape[2]).squeeze(2)
            pooled_outputs.append(pooled)

        # 5. Concatenate all features and apply dropout
        # Result: [Batch, n_filters * len(filter_sizes)]
        cat = self.dropout(torch.cat(pooled_outputs, dim=1))

        # 6. Final Classification
        return self.fc(cat)
