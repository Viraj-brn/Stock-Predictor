import torch
from src.model import CNN_RNN_AttnModel
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
base_closes = np.load("data/base_closes_test.npy")

# --- Attention Visualization ---
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

target_scaler = joblib.load("data/target_scaler.pkl")

# Inverse-transform to get actual returns
pred_returns = target_scaler.inverse_transform(y_pred_np)
actual_returns = target_scaler.inverse_transform(y_test_np)

# Convert returns to dollar prices using base close prices
pred_prices = base_closes * (1.0 + pred_returns)
actual_prices = base_closes * (1.0 + actual_returns)

# Calculate metrics per stock
mse = mean_squared_error(actual_prices, pred_prices, multioutput='raw_values')
mae = mean_absolute_error(actual_prices, pred_prices, multioutput='raw_values')

tickers = ["AAPL", "JNJ", "JPM", "MSFT", "PEP"]

print("--- Error Metrics (Dollar Prices) ---")
for i, ticker in enumerate(tickers):
    pct_err = mae[i] / actual_prices[:, i].mean() * 100
    print(f"{ticker} -> MSE: {mse[i]:.2f} | MAE: ${mae[i]:.2f} | MAPE: {pct_err:.1f}%")

# --- Subplot Visualization ---
plt.figure(figsize=(14, 10))
for i in range(5):
    plt.subplot(3, 2, i+1)
    plt.plot(actual_prices[:, i], label=f"Actual {tickers[i]}", linewidth=2)
    plt.plot(pred_prices[:, i], label=f"Predicted {tickers[i]}", linewidth=2, linestyle='--')
    plt.xlabel("Time Step")
    plt.ylabel("Close Price") 
    plt.title(f"{tickers[i]} - Predicted vs Actual Close Prices")
    plt.legend()

plt.tight_layout()
plt.show()