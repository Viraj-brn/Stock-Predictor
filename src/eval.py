import torch
from model import CNN_RNN_AttnModel
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_RNN_AttnModel()
model.load_state_dict(torch.load("models/best_model.pth"))
model.to(device)
model.eval()

X_test = torch.tensor(np.load("data/X_test.npy"), dtype=torch.float32)
y_test = torch.tensor(np.load("data/y_test.npy"), dtype=torch.float32)

with torch.no_grad():
    y_pred = model(X_test.to(device))

y_pred_np = y_pred.cpu().numpy()
y_test_np = y_test.cpu().numpy()
