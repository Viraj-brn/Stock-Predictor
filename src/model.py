import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_RNN_AttnModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN layer: learns local time-based patterns from 5 input features
        self.conv = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=5)
        
        # RNN layer (GRU): captures sequential patterns in the CNN output
        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        
        # Attention layer: lets the model focus on important time steps
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
        
        # Output layer: predicts 1 value (future Close price)
        self.fc = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()

    def forward(self,X):
        X = X.transpose(1, 2) #swap dim1 and dim2
        X = self.conv(X)
        X = self.relu(X)
        X = X.transpose(2, 1) #swap dim2 and dim1
        X, _ = self.gru(X) #output, hidden
        attn_output, _ = self.attn(X, X, X) #Query, key, value = X
        X = attn_output.mean(dim = 1)
        return self.fc(X)