import numpy as np
import pandas as pd
from typing import List
import itertools
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import os

from code import util

def save_model_and_scaler(model: Model, x_scaler, y_scaler=None, name: str = "lstm_model", subdir: str = "models/out/lstm"):
    """
    Save Keras model (.h5) and scalers (.pkl) under a subdirectory.

    Parameters:
        model (Model): Trained Keras model.
        x_scaler (StandardScaler): Feature scaler.
        y_scaler (StandardScaler or None): Target scaler (if used).
        name (str): Base filename (no extension).
        subdir (str): Directory to save files into. Default is models/out/lstm
    """
    os.makedirs(subdir, exist_ok=True)

    model_path = os.path.join(subdir, f"{name}.h5")
    scaler_path = os.path.join(subdir, f"{name}_scalers.pkl")

    # Save model
    model.save(model_path)

    # Save scalers
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "x_scaler": x_scaler,
            "y_scaler": y_scaler
        }, f)

    print(f"✅ Model saved to {model_path}")
    print(f"✅ Scalers saved to {scaler_path}")



def baseline(sequence_df: pd.DataFrame, epochs=50, batch_size=32):
    from code import util

    df = sequence_df.copy()
    df = df[df["X"].notna()].reset_index(drop=True)

    # Unpack data
    X = np.stack(df["X"].values)
    y = df["y"].values
    time_ids = df["time_id"].values
    start_times = df["start_time"].values

    # Standardize input features (reshape for scaler, then reshape back)
    n_steps, n_feats = X.shape[1], X.shape[2]
    X_reshaped = X.reshape(-1, n_feats)
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_reshaped).reshape(-1, n_steps, n_feats)

    # Sort and split by time_id (to avoid leakage)
    unique_ids = np.sort(np.unique(time_ids))
    split_idx = int(len(unique_ids) * 0.8)
    train_ids, test_ids = unique_ids[:split_idx], unique_ids[split_idx:]

    train_mask = np.isin(time_ids, train_ids)
    test_mask = np.isin(time_ids, test_ids)

    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_test, y_test = X_scaled[test_mask], y[test_mask]
    time_ids_test = time_ids[test_mask]
    start_times_test = start_times[test_mask]

    # Standardize targets
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Build model
    model = Sequential([
        Input(shape=X_train.shape[1:]),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse")

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    # Train
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_test, y_test_scaled),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1
    )

    # Predict and inverse-transform
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Evaluation
    mse = mean_squared_error(y_test, y_pred)
    print("=== Final Evaluation ===")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {np.sqrt(mse):.8f}")
    print(f"QLIKE: {util.qlike_loss(y_test, y_pred):.4f}")
    print(f"Directional Accuracy: {util.directional_accuracy(y_test, y_pred):.4f}")

    # Output DataFrame
    test_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "time_id": time_ids_test,
        "start_time": start_times_test
    })

    save_model_and_scaler(model, x_scaler, y_scaler, name="lstm_baseline")

    return model, history, test_df



def config_v256d03(sequence_df: pd.DataFrame, epochs=50, batch_size=32):
    from code import util

    df = sequence_df.copy()
    df = df[df["X"].notna()].reset_index(drop=True)

    # Unpack
    X = np.stack(df["X"].values)
    y = df["y"].values
    time_ids = df["time_id"].values
    start_times = df["start_time"].values

    # Standardize features
    n_steps, n_feats = X.shape[1], X.shape[2]
    X_reshaped = X.reshape(-1, n_feats)
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_reshaped).reshape(-1, n_steps, n_feats)

    # Split by time_id
    unique_ids = np.sort(np.unique(time_ids))
    split_idx = int(len(unique_ids) * 0.8)
    train_ids, test_ids = unique_ids[:split_idx], unique_ids[split_idx:]

    train_mask = np.isin(time_ids, train_ids)
    test_mask = np.isin(time_ids, test_ids)

    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_test, y_test = X_scaled[test_mask], y[test_mask]
    time_ids_test = time_ids[test_mask]
    start_times_test = start_times[test_mask]

    # Standardize targets
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Build model
    model = Sequential([
        Input(shape=X_train.shape[1:]),
        LSTM(256, return_sequences=False),
        Dropout(0.3),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse")

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    # Fit model
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_test, y_test_scaled),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1
    )

    # Predict
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print("=== Final Evaluation ===")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {np.sqrt(mse):.8f}")
    print(f"QLIKE: {util.qlike_loss(y_test, y_pred):.4f}")
    print(f"Directional Accuracy: {util.directional_accuracy(y_test, y_pred):.4f}")

    # Output for plotting
    test_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "time_id": time_ids_test,
        "start_time": start_times_test
    })

    save_model_and_scaler(model, x_scaler, y_scaler, name="config_v256d03")

    return model, history, test_df


def config_v256d03_more_feature(sequence_df: pd.DataFrame, epochs=50, batch_size=32):
    from code import util

    df = sequence_df.copy()
    df = df[df["X"].notna()].reset_index(drop=True)

    # Unpack
    X = np.stack(df["X"].values)
    y = df["y"].values
    time_ids = df["time_id"].values
    start_times = df["start_time"].values

    # Standardize features
    n_steps, n_feats = X.shape[1], X.shape[2]
    X_reshaped = X.reshape(-1, n_feats)
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_reshaped).reshape(-1, n_steps, n_feats)

    # Split by time_id
    unique_ids = np.sort(np.unique(time_ids))
    split_idx = int(len(unique_ids) * 0.8)
    train_ids, test_ids = unique_ids[:split_idx], unique_ids[split_idx:]

    train_mask = np.isin(time_ids, train_ids)
    test_mask = np.isin(time_ids, test_ids)

    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_test, y_test = X_scaled[test_mask], y[test_mask]
    time_ids_test = time_ids[test_mask]
    start_times_test = start_times[test_mask]

    # Standardize targets
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    # Build model
    model = Sequential([
        Input(shape=X_train.shape[1:]),
        LSTM(256, return_sequences=False),
        Dropout(0.3),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4), loss="mse")

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]

    # Fit model
    history = model.fit(
        X_train, y_train_scaled,
        validation_data=(X_test, y_test_scaled),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=1
    )

    # Predict
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    print("=== Final Evaluation ===")
    print(f"MSE: {mse:.8f}")
    print(f"RMSE: {np.sqrt(mse):.8f}")
    print(f"QLIKE: {util.qlike_loss(y_test, y_pred):.4f}")
    print(f"Directional Accuracy: {util.directional_accuracy(y_test, y_pred):.4f}")

    # Output for plotting
    test_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "time_id": time_ids_test,
        "start_time": start_times_test
    })

    save_model_and_scaler(model, x_scaler, y_scaler, name="config_v256d03_more_feature")

    return model, history, test_df
