import torch
import joblib
import sys
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(base_dir, 'src'))

from model import CNN_RNN_AttnModel

def load_model_and_scaler():
    model = CNN_RNN_AttnModel()
    
    model_path = os.path.join(base_dir, "models", "best_model.pth")
    scaler_path = os.path.join(base_dir, "data", "close_scaler.pkl")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    scaler = joblib.load(scaler_path)
    model.eval()
    
    return model, scaler