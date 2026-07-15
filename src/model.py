import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_RNN_AttnModel(nn.Module):
    """
    Improved CNN-GRU-Attention model for multi-stock price prediction.

    Architecture:
        Conv1d(25→64, k=3) → Conv1d(64→64, k=3) → LayerNorm → Dropout
        → 2-layer GRU(128) with dropout
        → 4-head MultiheadAttention with residual + LayerNorm
        → FC(128→64→5)

    ~180K trainable parameters (up from 40K in the original).
    """

    def __init__(self, n_features=25, n_outputs=5, dropout=0.2):
        super().__init__()

        # ── Convolutional feature extractor ──
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv_norm = nn.LayerNorm(64)
        self.conv_dropout = nn.Dropout(dropout)

        # ── Temporal encoder ──
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # ── Attention mechanism ──
        self.attn = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True,
            dropout=dropout,
        )
        self.attn_norm = nn.LayerNorm(128)

        # ── Output head ──
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )

        self.relu = nn.ReLU()

    def forward(self, X, return_attn=False):
        # X shape: (batch, seq_len, n_features)  e.g. (B, 60, 25)

        # ── CNN ──
        X = X.transpose(1, 2)               # → (B, 25, 60)
        X = self.relu(self.conv1(X))         # → (B, 64, 60)
        X = self.relu(self.conv2(X))         # → (B, 64, 60)
        X = X.transpose(1, 2)               # → (B, 60, 64)
        X = self.conv_norm(X)
        X = self.conv_dropout(X)

        # ── GRU ──
        X, _ = self.gru(X)                  # → (B, 60, 128)

        # ── Attention with residual connection ──
        attn_output, attn_weights = self.attn(X, X, X)
        X = self.attn_norm(X + attn_output)  # residual + layer norm

        # Take the last timestep (most recent info) instead of mean
        X = X[:, -1, :]                      # → (B, 128)

        # ── Output ──
        output = self.fc(X)                  # → (B, 5)

        if return_attn:
            return output, attn_weights
        return output