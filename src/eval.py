import torch
from model import CNN_RNN_AttnModel
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

sample = X_test[0].unsqueeze(0).to(device)

with torch.no_grad():
    _, attn_weights = model(sample, return_attn=True)

print("attn_weights shape:", attn_weights.shape)

attn_matrix = attn_weights.squeeze(0).cpu().numpy()
# print("attn_avg shape:", attn_avg.shape)
# print("attn_avg values:", attn_avg)
attn_per_timestep = attn_matrix[-1]
print("attn_per_timestep:", attn_per_timestep.shape)
print(attn_per_timestep)

plt.figure(figsize=(10, 4))
plt.plot(attn_per_timestep)
plt.title("Attention Weight per Input Timestep")
plt.xlabel("Timestep (0 = oldest, right = most recent)")
plt.ylabel("Attention")
plt.grid(True)
plt.tight_layout()
plt.show()

with torch.no_grad():
    y_pred = model(X_test.to(device))  # Don't pass return_attn=True here

y_pred_np = y_pred.cpu().numpy()
y_test_np = y_test.cpu().numpy()

scaler = joblib.load("data/close_scaler.pkl")
y_test_unscaled = scaler.inverse_transform(y_test_np.reshape(-1, 1))
y_pred_unscaled = scaler.inverse_transform(y_pred_np.reshape(-1, 1))

mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)

print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(y_test_unscaled, label="Actual", linewidth=2)
plt.plot(y_pred_unscaled, label="Predicted", linewidth=2)
plt.xlabel("Time Step")
plt.ylabel("Close Price (₹)")
plt.title("Predicted vs Actual Close Prices")
plt.legend()
plt.tight_layout()
plt.show()
