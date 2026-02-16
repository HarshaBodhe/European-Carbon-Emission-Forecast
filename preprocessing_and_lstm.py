import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# 1. Suppress TensorFlow/oneDNN logging BEFORE importing TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Cleaned up imports: Access through tf.keras for better editor stability
Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Input = tf.keras.layers.Input
EarlyStopping = tf.keras.callbacks.EarlyStopping

# Configuration
COUNTRIES = [
    "Germany", "France", "United Kingdom", "Italy", "Spain",
    "Poland", "Netherlands", "Belgium", "Sweden", "Austria"
]
TIME_STEPS = 12
MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_model(target_name, file_name, model_save_name, scaler_save_name):
    if not os.path.exists(file_name):
        print(f"❌ Missing dataset: {file_name}")
        return

    print(f"\n{'='*40}\n Training LSTM for {target_name}\n{'='*40}")

    df = pd.read_csv(file_name, parse_dates=['date']).sort_values("date").reset_index(drop=True)

    # Generate Lags
    for lag in [1, 3, 6, 12]:
        df[f"co2_lag_{lag}"] = df["co2_per_capita"].shift(lag)

    df = df.dropna().reset_index(drop=True)

    features = [
        "gdp_per_capita", "temp_avg", "renewable_share",
        "co2_lag_1", "co2_lag_3", "co2_lag_6", "co2_lag_12"
    ]

    # Fix scaling warning: Fit on DataFrame to keep feature names
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    joblib.dump(scaler, os.path.join(MODEL_DIR, scaler_save_name))

    X_seq, y_seq = create_sequences(df[features].values, df["co2_per_capita"].values)

    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Build Model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, activation="tanh", return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=16,
        validation_split=0.2,
        verbose=1,
        callbacks=[early]
    )

    model.save(os.path.join(MODEL_DIR, model_save_name))

    # Validation Plot
    plt.figure(figsize=(10, 4))
    preds = model.predict(X_test, verbose=0)
    plt.plot(y_test, label="Actual", color='blue')
    plt.plot(preds, label="Predicted", color='red', linestyle='--')
    plt.title(f"{target_name} CO₂ Forecast Validation")
    plt.legend()
    plt.savefig(os.path.join(MODEL_DIR, f"forecast_{target_name.lower()}.png"))
    plt.close()

    print(f"✅ Successfully saved {target_name} model and scaler.")

if __name__ == "__main__":
    # Train Berlin
    train_model("Berlin", "berlin_timeseries.csv", "lstm_berlin_model.h5", "berlin_scaler.joblib")

    # Train Countries
    for country in COUNTRIES:
        c_clean = country.lower().replace(' ', '_')
        train_model(country, f"{c_clean}_timeseries.csv", f"lstm_{c_clean}.h5", f"{c_clean}_scaler.joblib")