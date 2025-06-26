from flask import Flask, render_template
from model_loader import load_model_and_scaler
import numpy as np
import torch

model, scaler = load_model_and_scaler()
app = Flask(__name__)

@app.route('/')
def index():
    X_test = np.load("../data/X_test.npy")  # adjust if needed
    sample = torch.tensor(X_test[-1:]).float()  # last sample
    with torch.no_grad():
        y_pred = model(sample)
    y_pred_np = y_pred.numpy()
    predicted_price = scaler.inverse_transform(y_pred_np)[0][0]
    return render_template("index.html", prediction=round(predicted_price, 2))

if __name__ == '__main__':
    app.run(debug=True)
