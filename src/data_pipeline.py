import os
import numpy as np
import pandas as pd
from market_data import fetch_historical_ohlcv
from sklearn.preprocessing import StandardScaler
import joblib

def build_pipeline():
    # 1. Setup Data Directories
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    os.makedirs(base_dir, exist_ok=True)
    
    tickers = ["AAPL", "JNJ", "JPM", "MSFT", "PEP"]
    features = ['Close', 'High', 'Low', 'Open', 'Volume']
    
    print(f"Downloading data for {tickers}...")
    df = fetch_historical_ohlcv(tickers, start="2018-01-01")
    df = df.ffill().bfill()
    
    # 2. Convert raw prices to percentage returns
    # Returns are stationary (unlike prices which trend upward), so the model
    # can generalise to future price levels it has never seen during training.
    print("Computing percentage returns...")
    returns_df = df.pct_change()
    returns_df = returns_df.iloc[1:]   # drop the first NaN row
    df = df.iloc[1:]                   # keep aligned
    
    # 3. Build feature matrix from returns (num_days, 25)
    feature_list = []
    for ticker in tickers:
        for feature in features:
            col = returns_df[feature][ticker].values
            feature_list.append(col)
    X_returns = np.column_stack(feature_list)
    
    # Clean any inf/nan from pct_change (e.g. division by zero in Volume)
    X_returns = np.nan_to_num(X_returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 4. Build target: next-day close return for each ticker (num_days-1, 5)
    close_returns = returns_df['Close'][tickers].values   # (N, 5)
    y_returns = close_returns[1:]                          # shift: predict next day
    X_returns = X_returns[:-1]                             # align X with y
    
    # Save the raw close prices that correspond to each y sample.
    # base_close[i] is the close price on the LAST day of the input window
    # for sample i. At inference: predicted_price = base_close * (1 + pred_return)
    raw_closes = df['Close'][tickers].values[:-1]          # (N-1, 5)
    
    # Clean target
    y_returns = np.nan_to_num(y_returns, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 5. Scale features AND targets with StandardScaler
    # StandardScaler works much better than MinMaxScaler for returns because:
    #  - returns are roughly normally distributed around 0
    #  - no clipping to [0,1] means values outside training range still map reasonably
    print("Scaling features and targets with StandardScaler...")
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X_returns)
    
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y_returns)
    
    # Save both scalers
    joblib.dump(feature_scaler, os.path.join(base_dir, "feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(base_dir, "target_scaler.pkl"))
    print(f"Saved scalers to {base_dir}")
    
    # 6. Create sliding window sequences
    seq_length = 60
    X_seq, y_seq, base_closes_seq = [], [], []
    
    for i in range(len(X_scaled) - seq_length):
        X_seq.append(X_scaled[i : i + seq_length])
        y_seq.append(y_scaled[i + seq_length])
        # The base close for converting predicted return → price
        # is the close on the last day of this sequence (before the target day)
        base_closes_seq.append(raw_closes[i + seq_length])
        
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    base_closes_seq = np.array(base_closes_seq)
    
    print(f"Generated Sequences -> X: {X_seq.shape}, y: {y_seq.shape}, base_closes: {base_closes_seq.shape}")
    
    # 7. Train / Validation / Test Split (70% / 15% / 15%)
    train_size = int(len(X_seq) * 0.7)
    val_size = int(len(X_seq) * 0.15)
    
    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size : train_size + val_size], y_seq[train_size : train_size + val_size]
    X_test, y_test = X_seq[train_size + val_size:], y_seq[train_size + val_size:]
    base_closes_test = base_closes_seq[train_size + val_size:]
    
    # 8. Save tensors
    print("Saving datasets to disk...")
    np.save(os.path.join(base_dir, "X_train.npy"), X_train)
    np.save(os.path.join(base_dir, "y_train.npy"), y_train)
    np.save(os.path.join(base_dir, "X_val.npy"), X_val)
    np.save(os.path.join(base_dir, "y_val.npy"), y_val)
    np.save(os.path.join(base_dir, "X_test.npy"), X_test)
    np.save(os.path.join(base_dir, "y_test.npy"), y_test)
    np.save(os.path.join(base_dir, "base_closes_test.npy"), base_closes_test)
    
    print("Pipeline execution complete. Ready for training.")

if __name__ == "__main__":
    build_pipeline()