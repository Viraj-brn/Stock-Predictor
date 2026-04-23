import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_RNN_AttnModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 stocks * 5 features = 25 input channels
        self.conv = nn.Conv1d(in_channels=25, out_channels=32, kernel_size=5)
        
        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
        
        # Output layer predicts 5 values (1 future Close price for each stock)
        self.fc = nn.Linear(64, 5)
        self.relu = nn.ReLU()

    def forward(self, X, return_attn=False):
        X = X.transpose(1, 2) 
        X = self.conv(X)
        X = self.relu(X)
        X = X.transpose(2, 1) 
        X, _ = self.gru(X) 
        attn_output, attn_weights = self.attn(X, X, X) 
        X = attn_output.mean(dim=1)
        output = self.fc(X)
         
        if return_attn:
            return output, attn_weights
        return output