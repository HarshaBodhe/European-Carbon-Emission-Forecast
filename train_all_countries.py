# train_all_countries.py
"""
Train one LSTM per country in europe_timeseries.csv + Berlin.
Saves:
 - models/{country}_lstm.h5
 - models/{country}_scaler.joblib
 - results/{country}_forecast.png
 - results/metrics_summary.csv (aggregated)
Requires: tensorflow, pandas, numpy, matplotlib, scikit-learn
Run: python train_all_countries.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings
warnings.filterwarnings("ignore")

# params
DATA_FILE = "europe_timeseries.csv"
BERLIN_FILE = "berlin_timeseries.csv"
OUT_MODELS = "models"
OUT_RESULTS = "results"
SEQ_LEN = 12
EPOCHS = 80
BATCH = 16

os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_RESULTS, exist_ok=True)

def create_sequences(X, y, time_steps=SEQ_LEN):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, activation='tanh', input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    return {"mae": mae, "rmse": rmse, "mape": mape}

def train_single_country(df, country_name, is_berlin=False):
    """Train model for a single country/city"""
    print(f"=== Training for: {country_name}")
    
    # make sure series is long enough
    if len(df) < SEQ_LEN + 10:
        print(f"  skipping (too short): {country_name}")
        return None

    # create lag features
    for lag in [1,3,6,12]:
        df[f'co2_lag_{lag}'] = df['co2_per_capita'].shift(lag)
    df = df.dropna().reset_index(drop=True)

    # features and scaler
    feature_cols = ['gdp_per_capita','temp_avg','renewable_share'] + [f'co2_lag_{i}' for i in [1,3,6,12]]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].values)

    # save scaler
    country_key = "berlin" if is_berlin else country_name.lower().replace(' ','_')
    scaler_path = os.path.join(OUT_MODELS, f"{country_key}_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    X = df[feature_cols].values
    y = df['co2_per_capita'].values

    X_seq, y_seq = create_sequences(X, y, SEQ_LEN)
    # train/test split (time-based)
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # build & train
    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_split=0.15, epochs=EPOCHS, batch_size=BATCH, callbacks=[es], verbose=0)

    # predict & evaluate
    preds = model.predict(X_test, verbose=0).squeeze()
    m = evaluate(y_test, preds)
    print(f"  Done. RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, MAPE={m['mape']:.2f}%")

    # save model
    model_path = os.path.join(OUT_MODELS, f"{country_key}_lstm.h5")
    model.save(model_path)

    # plot actual vs pred (last N points)
    plt.figure(figsize=(8,4))
    t = df['date'].values[-len(y_test):]
    plt.plot(t, y_test, label='Actual')
    plt.plot(t, preds, label='Predicted')
    plt.xticks(rotation=30)
    plt.title(f"{country_name} - Actual vs Predicted CO2 per capita")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(OUT_RESULTS, f"{country_key}_forecast.png")
    plt.savefig(plot_path)
    plt.close()

    # return metrics
    return {
        "country": country_name,
        "rmse": m['rmse'],
        "mae": m['mae'],
        "mape": m['mape'],
        "model_path": model_path
    }

# Main execution
metrics_rows = []

# Train Berlin model first (for the app)
if os.path.exists(BERLIN_FILE):
    print("\n" + "="*60)
    print("Training Berlin Model")
    print("="*60)
    df_berlin = pd.read_csv(BERLIN_FILE, parse_dates=["date"])
    df_berlin = df_berlin.sort_values('date').reset_index(drop=True)
    result = train_single_country(df_berlin, "Berlin", is_berlin=True)
    if result:
        metrics_rows.append(result)
else:
    print(f"⚠️  {BERLIN_FILE} not found. Skipping Berlin model.")

# Train all countries from europe_timeseries.csv
if os.path.exists(DATA_FILE):
    print("\n" + "="*60)
    print("Training Country Models")
    print("="*60)
    df_all = pd.read_csv(DATA_FILE, parse_dates=["date"])
    countries = sorted(df_all['country'].unique())

    for country in countries:
        df = df_all[df_all['country'] == country].sort_values('date').reset_index(drop=True)
        result = train_single_country(df, country, is_berlin=False)
        if result:
            metrics_rows.append(result)
else:
    print(f"⚠️  {DATA_FILE} not found. Skipping country models.")

# write metrics summary
if metrics_rows:
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(OUT_RESULTS, "metrics_summary.csv"), index=False)
    print(f"\n✅ All done. Models -> {OUT_MODELS}, Results -> {OUT_RESULTS}")
    print(f"✅ Trained {len(metrics_rows)} models")
else:
    print("\n❌ No models were trained. Check data files.")
