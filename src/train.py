import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import CNN_RNN_AttnModel 

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data')
models_dir = os.path.join(base_dir, 'models')

os.makedirs(models_dir, exist_ok=True)

X_train = torch.tensor(np.load(os.path.join(data_dir, "X_train.npy")), dtype=torch.float32)
y_train = torch.tensor(np.load(os.path.join(data_dir, "y_train.npy")), dtype=torch.float32)
X_val = torch.tensor(np.load(os.path.join(data_dir, "X_val.npy")), dtype=torch.float32)
y_val = torch.tensor(np.load(os.path.join(data_dir, "y_val.npy")), dtype=torch.float32)
X_test = torch.tensor(np.load(os.path.join(data_dir, "X_test.npy")), dtype=torch.float32)
y_test = torch.tensor(np.load(os.path.join(data_dir, "y_test.npy")), dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = CNN_RNN_AttnModel().to(device)   
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.004)

num_epochs = 50
best_val_loss = float('inf')

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            val_loss += loss.item()
            
    val_loss /= len(val_loader)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:02d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(models_dir, "best_model.pth"))

print("Training complete! Best model saved to models/best_model.pth")