import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

import os
import pickle
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dropout, Bidirectional, LSTM,
    Attention, GlobalAveragePooling1D, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

import p1.util as util

def baseline(snapshot_df: pd.DataFrame,
                  basic_features: list = ['wap', 'log_return'],
                  epochs: int = 50,
                  batch_size: int = 32,
                  learning_rate: float = 1e-4,
                  es_patience: int = 10,
                  lr_patience: int = 5,
                  model_name: str = 'baseline'
                 ):
    """
    Train a simple LSTM baseline using basic features.
    Uses an 80%/20% train/validation split based on time_id.
    Includes EarlyStopping and ReduceLROnPlateau callbacks.
    During validation, records inference time per window.
    Saves model to out/lstm/{model_name}.h5 and scalers to out/lstm/{model_name}_scalers.pkl.
    Returns trained model, history, and validation DataFrame with 'inference_time'.
    """
    # Feature engineering
    seq_df = util.generate_tick_sequences(snapshot_df, feature_cols=basic_features)

    X = np.stack(seq_df['X'].values)
    y = seq_df['y'].values
    time_ids = seq_df['time_id'].values
    starts = seq_df['start_time'].values

    # 80/20 time_id split
    unique_ids = np.sort(np.unique(time_ids))
    split_idx = int(len(unique_ids) * 0.8)
    train_ids, val_ids = unique_ids[:split_idx], unique_ids[split_idx:]
    mask_train = np.isin(time_ids, train_ids)
    mask_val = np.isin(time_ids, val_ids)

    X_train, X_val = X[mask_train], X[mask_val]
    y_train, y_val = y[mask_train], y[mask_val]
    val_ids_out = time_ids[mask_val]
    val_starts_out = starts[mask_val]

    # Scale inputs & targets
    x_scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_s = x_scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_s = x_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    y_scaler = StandardScaler().fit(y_train.reshape(-1,1))
    y_train_s = y_scaler.transform(y_train.reshape(-1,1)).ravel()

    # Build model
    model = Sequential([Input(shape=X_train_s.shape[1:]), LSTM(128), Dense(1)])
    model.compile(optimizer=Adam(learning_rate), loss='mse')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=es_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, min_lr=1e-6)
    ]

    # Train
    history = model.fit(X_train_s, y_train_s,
                        validation_data=(X_val_s, y_val_s := y_scaler.transform(y_val.reshape(-1,1)).ravel()),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks, verbose=1)

    # Validation inference with timing
    inference_times = []
    y_pred_s = []
    for sample in X_val_s:
        start_t = time.perf_counter()
        pred = model.predict(sample[np.newaxis, ...], verbose=0).ravel()[0]
        inference_times.append(time.perf_counter() - start_t)
        y_pred_s.append(pred)
    y_pred_s = np.array(y_pred_s)
    y_pred = y_scaler.inverse_transform(y_pred_s.reshape(-1,1)).ravel()

    # Prepare validation DataFrame
    val_df = pd.DataFrame({
        'time_id': val_ids_out,
        'start_time': val_starts_out,
        'y_true': y_val,
        'y_pred': y_pred,
        'inference_time': inference_times
    })

    # Metrics
    mse = np.mean((y_val - y_pred)**2)
    avg_time = np.mean(inference_times)
    print(f"Validation MSE: {mse:.8f}")
    print(f"Validation RMSE: {np.sqrt(mse):.8f}")
    print(f"Validation QLIKE: {util.qlike_loss(y_val, y_pred):.4f}")
    print(f"Directional Acc: {util.directional_accuracy(y_val, y_pred):.4f}")
    print(f"Average inference time per window: {avg_time:.6f} seconds")

    # Save model & scalers
    out_dir = 'out/lstm'
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f'{model_name}.h5'), save_format='tf')
    with open(os.path.join(out_dir, f'{model_name}_scalers.pkl'), 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)
    print(f"Model & scalers saved to {out_dir}")

    return model, history, val_df




def advanced(snapshot_df: pd.DataFrame,
               basic_features: list,
               epochs: int = 100,
               batch_size: int = 64,
               learning_rate: float = 1e-4,
               es_patience: int = 15,
               lr_patience: int = 7,
               model_name: str = 'advanced'):
    """
    advanced LSTM-based volatility forecaster:
      - 1D-CNN layers for local pattern extraction
      - BiLSTM layers stacked for long/short-term memory
      - Self-attention to focus on key timesteps
      - Global pooling + Dense head
      - Custom QLIKE loss option
    """
    # feature engineering
    seq_df = util.generate_tick_sequences(snapshot_df, feature_cols=basic_features)

    X = np.stack(seq_df['X'].values)
    y = seq_df['y'].values
    time_ids = seq_df['time_id'].values
    starts = seq_df['start_time'].values

    # 80/20 split by time_id
    unique_ids = np.sort(np.unique(time_ids))
    split = int(len(unique_ids) * 0.8)
    train_ids, val_ids = unique_ids[:split], unique_ids[split:]
    mask_train = np.isin(time_ids, train_ids)
    mask_val = np.isin(time_ids, val_ids)
    X_train, X_val = X[mask_train], X[mask_val]
    y_train, y_val = y[mask_train], y[mask_val]
    val_ids_out = time_ids[mask_val]
    val_starts_out = starts[mask_val]

    # scaling
    x_scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    X_train_s = x_scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_s = x_scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    y_train_s = y_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()

    # build model
    W, D = X_train_s.shape[1], X_train_s.shape[2]
    inp = Input(shape=(W, D))
    x = Conv1D(32, 3, padding='same', activation='relu')(inp)
    x = Conv1D(64, 3, padding='same', activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.3)(x)
    # self-attention
    attn_out = Attention()([x, x])
    # pool & head
    x = GlobalAveragePooling1D()(attn_out)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(1, activation='linear')(x)
    model = Model(inp, out)

    # compile with QLIKE
    model.compile(
        optimizer=Adam(learning_rate),
        loss='mse',
        metrics=[]
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=es_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_patience, min_lr=1e-6)
    ]

    # train
    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # inference timing
    times, preds = [], []
    for s in X_val_s:
        t0 = time.perf_counter()
        p = model.predict(s[np.newaxis, ...], verbose=0).ravel()[0]
        times.append(time.perf_counter() - t0)
        preds.append(p)
    preds = np.array(preds)
    y_pred = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()

    val_df = pd.DataFrame({
        'time_id': val_ids_out,
        'start_time': val_starts_out,
        'y_true': y_val,
        'y_pred': y_pred,
        'inference_time': times
    })

    # evaluation
    mse = np.mean((y_val - y_pred) ** 2)
    print(f"Final Model MSE: {mse:.8f}")
    print(f"Final Model RMSE: {np.sqrt(mse):.8f}")
    print(f"Final Model QLIKE: {util.qlike_loss(y_val, y_pred):.4f}")
    print(f"Directional Acc: {util.directional_accuracy(y_val, y_pred):.4f}")
    print(f"Avg Inference Time: {np.mean(times):.6f}s")

    # save
    out_dir = 'out/lstm'
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f'{model_name}.h5'), save_format='tf')
    with open(os.path.join(out_dir, f'{model_name}_scalers.pkl'), 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)
    print(f"Saved final model & scalers to {out}")

    return model, history, val_df
