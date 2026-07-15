import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import CNN_RNN_AttnModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_dir = os.path.join(base_dir, 'data')
models_dir = os.path.join(base_dir, 'models')

os.makedirs(models_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
X_train = torch.tensor(np.load(os.path.join(data_dir, "X_train.npy")), dtype=torch.float32)
y_train = torch.tensor(np.load(os.path.join(data_dir, "y_train.npy")), dtype=torch.float32)
X_val = torch.tensor(np.load(os.path.join(data_dir, "X_val.npy")), dtype=torch.float32)
y_val = torch.tensor(np.load(os.path.join(data_dir, "y_val.npy")), dtype=torch.float32)
X_test = torch.tensor(np.load(os.path.join(data_dir, "X_test.npy")), dtype=torch.float32)
y_test = torch.tensor(np.load(os.path.join(data_dir, "y_test.npy")), dtype=torch.float32)

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 150
PATIENCE = 20          # early stopping patience
GRAD_CLIP = 1.0        # gradient clipping max norm
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10

# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

val_data = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# ---------------------------------------------------------------------------
# Model, optimizer, scheduler
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = CNN_RNN_AttnModel().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=SCHEDULER_FACTOR,
    patience=SCHEDULER_PATIENCE,
)

# ---------------------------------------------------------------------------
# Training loop with early stopping
# ---------------------------------------------------------------------------
best_val_loss = float('inf')
epochs_without_improvement = 0

print(f"\nStarting training (max {NUM_EPOCHS} epochs, early stopping patience={PATIENCE})...\n")

for epoch in range(NUM_EPOCHS):
    # ── Train ──
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in GRU
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ── Validate ──
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # ── LR Scheduler ──
    scheduler.step(val_loss)

    # ── Logging ──
    current_lr = optimizer.param_groups[0]['lr']
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.6f}")

    # ── Early stopping ──
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), os.path.join(models_dir, "best_model.pth"))
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

print(f"\nTraining complete! Best val loss: {best_val_loss:.6f}")
print(f"Model saved to models/best_model.pth")