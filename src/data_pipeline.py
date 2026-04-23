import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import joblib

def build_pipeline():
    # 1. Setup Data Directories
    # Adjust paths if you are running this from the root directory instead of /src
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(base_dir, exist_ok=True)
    
    tickers = ["JNJ", "JPM", "MSFT", "PEP", "XOM"]
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    print(f"Downloading data for {tickers}...")
    # Fetching ~6 years of data to give the model plenty of training context
    df = yf.download(tickers, start="2018-01-01", end="2024-04-20")
    
    # Clean missing data (forward fill, then backward fill)
    df = df.ffill().bfill()

    # 2. Extract and Align Features
    # We flatten the multi-index dataframe so each day has 25 features in a strict order:
    # [JNJ_Close, JNJ_High..., JPM_Close, JPM_High..., etc.]
    feature_list = []
    for ticker in tickers:
        for feature in features:
            feature_list.append(df[feature][ticker].values)
            
    # X_raw shape: (num_days, 25)
    X_raw = np.column_stack(feature_list)
    
    # y_raw shape: (num_days, 5) - We only want to predict the Close prices
    y_raw = df['Close'][tickers].values
    
    # 3. Scaling
    print("Scaling features and targets...")
    feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(X_raw)
    
    # We use a separate scaler for 'y' so we can inverse_transform the predictions later in app.py
    close_scaler = MinMaxScaler()
    y_scaled = close_scaler.fit_transform(y_raw)
    
    # Save the close_scaler for the Flask backend
    scaler_path = os.path.join(base_dir, "close_scaler.pkl")
    joblib.dump(close_scaler, scaler_path)
    
    # 4. Create Sliding Window Sequences
    seq_length = 60 # Using 60 days of history to predict the next day
    X_seq, y_seq = [], []
    
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i : i + seq_length])
        y_seq.append(y_scaled[i + seq_length])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"Generated Sequences -> X shape: {X_seq.shape}, y shape: {y_seq.shape}")
    
    # 5. Train / Validation / Test Split (70% / 15% / 15%)
    train_size = int(len(X_seq) * 0.7)
    val_size = int(len(X_seq) * 0.15)
    
    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size : train_size + val_size], y_seq[train_size : train_size + val_size]
    X_test, y_test = X_seq[train_size + val_size:], y_seq[train_size + val_size:]
    
    # 6. Save Tensors
    print("Saving datasets to disk...")
    np.save(os.path.join(base_dir, "X_train.npy"), X_train)
    np.save(os.path.join(base_dir, "y_train.npy"), y_train)
    np.save(os.path.join(base_dir, "X_val.npy"), X_val)
    np.save(os.path.join(base_dir, "y_val.npy"), y_val)
    np.save(os.path.join(base_dir, "X_test.npy"), X_test)
    np.save(os.path.join(base_dir, "y_test.npy"), y_test)
    
    print("Pipeline execution complete. Ready for training.")

if __name__ == "__main__":
    build_pipeline()