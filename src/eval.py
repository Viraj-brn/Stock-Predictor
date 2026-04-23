import torch
from src.model import CNN_RNN_AttnModel # adjust import path if needed
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_RNN_AttnModel()
model.load_state_dict(torch.load("models/best_model.pth"))
model.to(device)
model.eval()

X_test = torch.tensor(np.load("data/X_test.npy"), dtype=torch.float32)
y_test = torch.tensor(np.load("data/y_test.npy"), dtype=torch.float32)

# --- Attention Visualization (Works exactly the same) ---
sample = X_test[0].unsqueeze(0).to(device)

with torch.no_grad():
    _, attn_weights = model(sample, return_attn=True)

print("attn_weights shape:", attn_weights.shape)

attn_matrix = attn_weights.squeeze(0).cpu().numpy()
attn_per_timestep = attn_matrix[-1]
print("attn_per_timestep:", attn_per_timestep.shape)

plt.figure(figsize=(10, 4))
plt.plot(attn_per_timestep)
plt.title("Attention Weight per Input Timestep")
plt.xlabel("Timestep (0 = oldest, right = most recent)")
plt.ylabel("Attention")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Prediction & Evaluation ---
with torch.no_grad():
    y_pred = model(X_test.to(device))  

y_pred_np = y_pred.cpu().numpy()
y_test_np = y_test.cpu().numpy()

scaler = joblib.load("data/close_scaler.pkl")

# REMOVED .reshape(-1, 1) because the data already has 5 columns
y_test_unscaled = scaler.inverse_transform(y_test_np)
y_pred_unscaled = scaler.inverse_transform(y_pred_np)

# Calculate metrics per stock instead of an average mess
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled, multioutput='raw_values')
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled, multioutput='raw_values')

tickers = ["JNJ", "JPM", "MSFT", "PEP", "XOM"]

print("--- Error Metrics ---")
for i, ticker in enumerate(tickers):
    print(f"{ticker} -> MSE: {mse[i]:.2f} | MAE: {mae[i]:.2f}")

# --- Subplot Visualization with Proper Named Labels ---
plt.figure(figsize=(14, 10))
for i in range(5):
    plt.subplot(3, 2, i+1) # 3 rows, 2 columns of plots
    plt.plot(y_test_unscaled[:, i], label=f"Actual {tickers[i]}", linewidth=2)
    plt.plot(y_pred_unscaled[:, i], label=f"Predicted {tickers[i]}", linewidth=2, linestyle='--')
    plt.xlabel("Time Step")
    plt.ylabel("Close Price") 
    plt.title(f"{tickers[i]} - Predicted vs Actual Close Prices")
    plt.legend()

plt.tight_layout()
plt.show()