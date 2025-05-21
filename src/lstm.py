# Standard library
import os
import time
import pickle
from typing import List

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense,
    Conv1D, Dropout,
    Bidirectional, Attention,
    GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler

import os
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Bidirectional, LSTM, Dropout,
    Dense, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler


# Local modules
import src.util as util


def baseline(snapshot_df: pd.DataFrame,
                    basic_features: list = ['wap', 'log_return'],
                    epochs: int = 50,
                    batch_size: int = 16,
                    learning_rate: float = 1e-4,
                    es_patience: int = 10,
                    lr_patience: int = 5,
                    model_name: str = 'baseline_stream'):
    """
    Streaming LSTM baseline using data_generator:
      - 80/20 split on time_id, per-batch loading to save memory
      - Same architecture: single LSTM(128) + Dense
    """
    seq_df = util.generate_tick_sequences(snapshot_df, feature_cols=basic_features)
    tids = seq_df['time_id'].values
    unique_t = np.sort(np.unique(tids))
    split = int(len(unique_t) * 0.8)
    train_ids, val_ids = unique_t[:split], unique_t[split:]
    train_df = seq_df[seq_df['time_id'].isin(train_ids)].reset_index(drop=True)
    val_df = seq_df[seq_df['time_id'].isin(val_ids)].reset_index(drop=True)

    flat_X = np.vstack(train_df['X'].values).astype('float32')
    x_scaler = StandardScaler().fit(flat_X)
    y_scaler = StandardScaler().fit(train_df['y'].values.reshape(-1,1))

    train_gen = data_generator(train_df, x_scaler, y_scaler, batch_size)
    val_gen   = data_generator(val_df,   x_scaler, y_scaler, batch_size)
    steps_per_epoch  = len(train_df) // batch_size
    validation_steps = len(val_df)   // batch_size

    W, D = train_df.iloc[0]['X'].shape
    inp = Input((W, D))
    x = LSTM(128)(inp)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(learning_rate), loss='mse')

    callbacks = [
        EarlyStopping('val_loss', patience=es_patience, restore_best_weights=True),
        ReduceLROnPlateau('val_loss', factor=0.5, patience=lr_patience, min_lr=1e-6)
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    inference_times, preds = [], []
    flat_val = np.vstack(val_df['X'].values).astype('float32')
    X_val_norm = x_scaler.transform(flat_val).reshape(-1, W, D)
    y_val = val_df['y'].values
    for seq in X_val_norm:
        t0 = time.perf_counter()
        p = model.predict(seq[np.newaxis,...], verbose=0).ravel()[0]
        inference_times.append(time.perf_counter() - t0)
        preds.append(p)
    preds = np.array(preds)
    y_pred = y_scaler.inverse_transform(preds.reshape(-1,1)).ravel()

    results = val_df[['time_id','start_time']].copy()
    results['y_true'] = y_val
    results['y_pred'] = y_pred
    results['inference_time'] = inference_times

    mse = np.mean((results['y_true'] - results['y_pred'])**2)
    print(f"Validation MSE: {mse:.8f}")
    print(f"Validation RMSE: {np.sqrt(mse):.8f}")
    print(f"Validation QLIKE: {util.qlike_loss(results['y_true'], results['y_pred']):.4f}")
    print(f"Dir Acc: {util.directional_accuracy(results['y_true'], results['y_pred']):.4f}")

    out_dir = 'out/lstm'
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f'{model_name}.h5'), save_format='tf')
    with open(os.path.join(out_dir, f'{model_name}_scalers.pkl'), 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)
    print(f"Model & scalers saved to {out_dir}")

    return model, history, results


def data_generator(seq_df, x_scaler, y_scaler, batch_size):
    n = len(seq_df)
    while True:
        seq_df = seq_df.sample(frac=1).reset_index(drop=True)
        for i in range(0, n, batch_size):
            batch = seq_df.iloc[i:i+batch_size]
            Xb = np.stack(batch['X'].values).astype('float32')
            flat = Xb.reshape(-1, Xb.shape[-1])
            Xb_s = x_scaler.transform(flat).reshape(Xb.shape)
            yb = batch['y'].values.astype('float32').reshape(-1,1)
            yb_s = y_scaler.transform(yb).ravel()
            yield Xb_s, yb_s


def weighted_mse(y_true, y_pred):
    abs_y = tf.abs(y_true)
    denom = tf.reduce_mean(abs_y) + 1e-6
    w = 1.0 + abs_y / denom
    return tf.reduce_mean(w * tf.square(y_true - y_pred))


def baseline_weighted_mse(snapshot_df: pd.DataFrame,
                    basic_features: list = ['wap', 'log_return'],
                    epochs: int = 50,
                    batch_size: int = 16,
                    learning_rate: float = 1e-4,
                    es_patience: int = 10,
                    lr_patience: int = 5,
                    model_name: str = 'baseline_weighted_mse'):
    """
    Streaming LSTM baseline using data_generator:
      - 80/20 split on time_id, per-batch loading to save memory
      - Same architecture: single LSTM(128) + Dense
    """
    seq_df = util.generate_tick_sequences(snapshot_df, feature_cols=basic_features)
    tids = seq_df['time_id'].values
    unique_t = np.sort(np.unique(tids))
    split = int(len(unique_t) * 0.8)
    train_ids, val_ids = unique_t[:split], unique_t[split:]
    train_df = seq_df[seq_df['time_id'].isin(train_ids)].reset_index(drop=True)
    val_df = seq_df[seq_df['time_id'].isin(val_ids)].reset_index(drop=True)

    flat_X = np.vstack(train_df['X'].values).astype('float32')
    x_scaler = StandardScaler().fit(flat_X)
    y_scaler = StandardScaler().fit(train_df['y'].values.reshape(-1,1))

    train_gen = data_generator(train_df, x_scaler, y_scaler, batch_size)
    val_gen   = data_generator(val_df,   x_scaler, y_scaler, batch_size)
    steps_per_epoch  = len(train_df) // batch_size
    validation_steps = len(val_df)   // batch_size

    W, D = train_df.iloc[0]['X'].shape
    inp = Input((W, D))
    x = LSTM(128)(inp)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(learning_rate), loss=weighted_mse)

    callbacks = [
        EarlyStopping('val_loss', patience=es_patience, restore_best_weights=True),
        ReduceLROnPlateau('val_loss', factor=0.5, patience=lr_patience, min_lr=1e-6)
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    inference_times, preds = [], []
    flat_val = np.vstack(val_df['X'].values).astype('float32')
    X_val_norm = x_scaler.transform(flat_val).reshape(-1, W, D)
    y_val = val_df['y'].values
    for seq in X_val_norm:
        t0 = time.perf_counter()
        p = model.predict(seq[np.newaxis,...], verbose=0).ravel()[0]
        inference_times.append(time.perf_counter() - t0)
        preds.append(p)
    preds = np.array(preds)
    y_pred = y_scaler.inverse_transform(preds.reshape(-1,1)).ravel()

    results = val_df[['time_id','start_time']].copy()
    results['y_true'] = y_val
    results['y_pred'] = y_pred
    results['inference_time'] = inference_times

    mse = np.mean((results['y_true'] - results['y_pred'])**2)
    print(f"Validation MSE: {mse:.8f}")
    print(f"Validation RMSE: {np.sqrt(mse):.8f}")
    print(f"Validation QLIKE: {util.qlike_loss(results['y_true'], results['y_pred']):.4f}")
    print(f"Dir Acc: {util.directional_accuracy(results['y_true'], results['y_pred']):.4f}")

    out_dir = 'out/lstm'
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f'{model_name}.h5'), save_format='tf')
    with open(os.path.join(out_dir, f'{model_name}_scalers.pkl'), 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)
    print(f"Model & scalers saved to {out_dir}")

    return model, history, results


import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler



def baseline_two_stage_eval(snapshot_df: pd.DataFrame,
             basic_features: list = ['wap', 'log_return'],
             epochs_stage1: int = 30,
             epochs_stage2: int = 20,
             batch_size: int = 16,
             learning_rate: float = 1e-4,
             es_patience: int = 10,
             lr_patience: int = 5,
             model_name: str = '2_eval'):
    """
    Two-stage LSTM training:
      1. Stage 1: train on MSE to learn the smooth trend.
      2. Stage 2: fine-tune on QLIKE (or weighted MSE) to capture spikes.
    Returns final model, stage2 history, and results DataFrame.
    """
    seq_df = util.generate_tick_sequences(snapshot_df, feature_cols=basic_features)
    tids = np.sort(seq_df['time_id'].unique())
    split = int(len(tids) * 0.8)
    train_df = seq_df[seq_df['time_id'].isin(tids[:split])].reset_index(drop=True)
    val_df   = seq_df[seq_df['time_id'].isin(tids[split:])].reset_index(drop=True)

    flat_X = np.vstack(train_df['X'].values).astype('float32')
    x_scaler = StandardScaler().fit(flat_X)
    y_scaler = StandardScaler().fit(train_df['y'].values.reshape(-1,1))

    def gen(df):
        n = len(df)
        while True:
            df = df.sample(frac=1).reset_index(drop=True)
            for i in range(0, n, batch_size):
                batch = df.iloc[i:i+batch_size]
                Xb = np.stack(batch['X'].values).astype('float32')
                Xb_s = x_scaler.transform(Xb.reshape(-1, Xb.shape[-1])).reshape(Xb.shape)
                yb = batch['y'].values.reshape(-1,1).astype('float32')
                yb_s = y_scaler.transform(yb).ravel()
                yield Xb_s, yb_s

    steps_per_epoch  = len(train_df) // batch_size
    validation_steps = len(val_df)   // batch_size
    train_gen = gen(train_df)
    val_gen   = gen(val_df)

    W, D = train_df.iloc[0]['X'].shape


    def quantile_loss(q):
        def loss(y, y_hat):
            e = y - y_hat
            return K.mean(K.maximum(q*e, (q-1)*e))
        return loss


    inp = Input((W, D))
    x   = LSTM(128)(inp)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(
        optimizer=Adam(learning_rate),
        loss=weighted_mse,
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    ckpt1 = ModelCheckpoint(f'{model_name}_stage1.h5',
                             save_best_only=True, monitor='val_loss')
    es1   = EarlyStopping('val_loss', patience=es_patience, restore_best_weights=False)
    rlr1  = ReduceLROnPlateau('val_loss', factor=0.5,
                              patience=lr_patience, min_lr=1e-6)

    model.fit(
        train_gen,
        epochs=epochs_stage1,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[ckpt1, es1, rlr1],
        verbose=1
    )

    model.load_weights(f'{model_name}_stage1.h5')
    model.compile(
        optimizer=Adam(learning_rate * 0.1),
        loss=quantile_loss(0.7),
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    ckpt2 = ModelCheckpoint(f'{model_name}_stage2.h5',
                             save_best_only=True, monitor='val_loss')
    es2   = EarlyStopping('val_loss', patience=max(5, es_patience//2),
                          restore_best_weights=True)
    rlr2  = ReduceLROnPlateau('val_loss', factor=0.5,
                              patience=max(3, lr_patience//2), min_lr=1e-7)

    history2 = model.fit(
        train_gen,
        epochs=epochs_stage2,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=[ckpt2, es2, rlr2],
        verbose=1
    )

    model.load_weights(f'{model_name}_stage2.h5')

    flat_val = np.vstack(val_df['X'].values).astype('float32')
    X_val_norm = x_scaler.transform(flat_val).reshape(-1, W, D)
    y_true = val_df['y'].values
    preds_s, times = [], []
    for seq in X_val_norm:
        t0 = time.perf_counter()
        p  = model.predict(seq[np.newaxis], verbose=0).ravel()[0]
        times.append(time.perf_counter() - t0)
        preds_s.append(p)
    preds = y_scaler.inverse_transform(np.array(preds_s).reshape(-1,1)).ravel()

    results = val_df[['time_id','start_time']].copy()
    results['y_true']      = y_true
    results['y_pred']      = preds
    results['inference_time'] = times

    mse = np.mean((results['y_true'] - results['y_pred'])**2)
    print(f"Final RMSE:  {np.sqrt(mse):.8f}")
    print(f"Final QLIKE:{util.qlike_loss(results['y_true'], results['y_pred']):.4f}")
    print(f"Dir Acc:    {util.directional_accuracy(results['y_true'], results['y_pred']):.4f}")

    out_dir = 'out/lstm'
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f'{model_name}.h5'), save_format='tf')
    with open(os.path.join(out_dir, f'{model_name}_scalers.pkl'), 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

    return model, history2, results





import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_recall_curve, mean_squared_error


class WarmUp(tf.keras.callbacks.Callback):
    def __init__(self, warmup_steps, target_lr):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr

    def on_train_batch_begin(self, batch, logs=None):
        it = self.model.optimizer.iterations.numpy()
        if it < self.warmup_steps:
            warmup_lr = self.target_lr * (it / float(self.warmup_steps))
            try:
                self.model.optimizer.learning_rate = warmup_lr
            except AttributeError:
                self.model.optimizer._set_hyper('learning_rate', warmup_lr)





def baseline_moe_spike(
    snapshot_df: pd.DataFrame,
    basic_features: list,
    spike_quantile: float = 0.95,
    target_frac: float = 0.20,
    min_factor: float = 1.0,
    max_factor: float = 5.0,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    es_patience: int = 15,
    lr_patience: int = 7,
    model_name: str = 'moe_spike'
):

    seq_df = util.generate_tick_sequences(snapshot_df, feature_cols=basic_features)
    tids = np.sort(seq_df['time_id'].unique())
    split = int(len(tids) * 0.8)
    train_df = seq_df[seq_df['time_id'].isin(tids[:split])].reset_index(drop=True)
    val_df   = seq_df[seq_df['time_id'].isin(tids[split:])].reset_index(drop=True)

    spike_thresh = train_df['y'].quantile(spike_quantile)
    train_df['spike'] = (train_df['y'] > spike_thresh).astype(int)
    val_df['spike']   = (val_df['y']   > spike_thresh).astype(int)

    flat_X = np.vstack(train_df['X'].values).astype('float32')
    x_scaler = StandardScaler().fit(flat_X)
    y_scaler = StandardScaler().fit(train_df['y'].values.reshape(-1,1))

    pos = train_df['spike'].sum()
    neg = len(train_df) - pos
    raw_factor = (target_frac * neg / (pos * (1 - target_frac))) if pos>0 else 1.0
    oversample_factor = int(np.clip(raw_factor, min_factor, max_factor))
    print(f"Dynamic oversample_factor = {oversample_factor:.1f}")

    def generator(df):
        pos_df = df[df['spike']==1]
        neg_df = df[df['spike']==0]
        while True:
            neg_shuf = neg_df.sample(frac=1).reset_index(drop=True)
            pos_shuf = pos_df.sample(frac=oversample_factor, replace=True).reset_index(drop=True)
            aug = pd.concat([neg_shuf, pos_shuf]).sample(frac=1).reset_index(drop=True)
            for i in range(0, len(aug), batch_size):
                batch = aug.iloc[i:i+batch_size]
                Xb = np.stack(batch['X'].values).astype('float32')
                Xb_s = x_scaler.transform(Xb.reshape(-1, Xb.shape[-1])).reshape(Xb.shape)
                yb = batch['y'].values.reshape(-1,1).astype('float32')
                yb_s = y_scaler.transform(yb).ravel()
                spk = batch['spike'].values.astype('float32').reshape(-1,1)
                yield Xb_s, {'vol': yb_s, 'spike': spk}

    def val_generator(df, batch_size, x_scaler, y_scaler):
        W,D = df.iloc[0]['X'].shape
        while True:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                Xb = np.stack(batch['X'].values).astype('float32')
                Xb = x_scaler.transform(Xb.reshape(-1,D)).reshape(-1,W,D)
                yb = batch['y'].values.reshape(-1,1).astype('float32')
                yb = y_scaler.transform(yb).ravel()
                spk = batch['spike'].values.astype('float32').reshape(-1,1)
                yield Xb, {'vol': yb, 'spike': spk}

    steps  = len(train_df) // batch_size
    vsteps = int(np.ceil(len(val_df) / batch_size))
    train_gen = generator(train_df)
    val_gen   = val_generator(val_df, batch_size, x_scaler, y_scaler)
    W, D = train_df.iloc[0]['X'].shape

    inp = Input((W, D))
    h = LSTM(128)(inp)
    spike_prob   = Dense(1, activation='sigmoid', name='spike')(h)
    expert_norm  = Dense(1, name='expert_normal')(h)
    expert_spike = Dense(1, name='expert_spike')(h)
    vol_out = Lambda(lambda args: (1-args[0])*args[1] + args[0]*args[2], name='vol')([
        spike_prob, expert_norm, expert_spike
    ])
    model = Model(inp, [vol_out, spike_prob])

    def weighted_mse(y_true, y_pred):
        abs_y = K.abs(y_true)
        w = 1.0 + abs_y / (K.mean(abs_y) + K.epsilon())
        return K.mean(w * K.square(y_true - y_pred))

    def focal_bce(gamma=4.0, alpha=0.25):
        bce = tf.keras.losses.BinaryCrossentropy()
        def loss(y_true, y_pred):
            b = bce(y_true, y_pred)
            p_t = y_true*y_pred + (1-y_true)*(1-y_pred)
            return K.mean(alpha * K.pow(1-p_t, gamma) * b)
        return loss

    model.compile(
        optimizer=Adam(learning_rate),
        loss={'vol': weighted_mse, 'spike': focal_bce()},
        loss_weights={'vol':1.0, 'spike':0.5},
        metrics={'vol': tf.keras.metrics.RootMeanSquaredError(name='rmse'),
                 'spike': 'accuracy'}
    )

    total_train_steps = steps * epochs
    warmup_steps = int(0.1 * total_train_steps)
    warmup_cb = WarmUp(warmup_steps, learning_rate)
    ckpt = ModelCheckpoint(f'{model_name}.h5', save_best_only=True,
                           monitor='val_vol_loss', mode='min')
    es   = EarlyStopping(monitor='val_vol_loss', mode='min',
                         patience=es_patience, restore_best_weights=True)
    rlr  = ReduceLROnPlateau(monitor='val_vol_loss', mode='min',
                             factor=0.5, patience=lr_patience, min_lr=1e-7)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=steps,
        validation_steps=vsteps,
        epochs=epochs,
        callbacks=[warmup_cb, ckpt, es, rlr],
        verbose=1
    )

    inf_model = Model(inp, [vol_out, spike_prob, expert_norm, expert_spike])
    inf_model.load_weights(f'{model_name}.h5')

    flat_val = np.vstack(val_df['X'].values).astype('float32')
    Xv = x_scaler.transform(flat_val).reshape(-1, W, D)
    y_true = val_df['y'].values
    vols, probs, normals, spikes_ex, times = [], [], [], [], []
    for seq in Xv:
        t0 = time.perf_counter()
        v, p, n_ex, s_ex = inf_model.predict(seq[np.newaxis], verbose=0)
        times.append(time.perf_counter() - t0)
        vols.append(v.ravel()[0])
        probs.append(p.ravel()[0])
        normals.append(n_ex.ravel()[0])
        spikes_ex.append(s_ex.ravel()[0])

    results = val_df[['time_id','start_time']].copy()
    results['y_true']        = y_true
    results['y_pred']        = y_scaler.inverse_transform(np.array(vols).reshape(-1,1)).ravel()
    results['spike_prob']    = probs
    results['expert_normal'] = y_scaler.inverse_transform(np.array(normals).reshape(-1,1)).ravel()
    results['expert_spike']  = y_scaler.inverse_transform(np.array(spikes_ex).reshape(-1,1)).ravel()
    results['inference_time'] = times

    labels = val_df['spike'].values
    prec, rec, th = precision_recall_curve(labels, probs)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_thr = th[np.argmax(f1)]
    print(f"Best spike threshold by F1: {best_thr:.3f}")
    mask = np.array(probs) > best_thr
    results['y_pred_calibrated'] = (~mask).astype(float)*results['expert_normal'] + mask.astype(float)*results['expert_spike']

    mse = mean_squared_error(results['y_true'], results['y_pred_calibrated'])
    print(f"Calibrated RMSE: {np.sqrt(mse):.8f}")
    print(f"Calibrated QLIKE: {util.qlike_loss(results['y_true'], results['y_pred_calibrated']):.4f}")
    spike_pred_bin = mask.astype(int)
    print("Spike Acc:", accuracy_score(labels, spike_pred_bin))
    print("Spike Recall:", recall_score(labels, spike_pred_bin))

    out_dir = 'out/lstm'
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f'{model_name}_final.h5'), save_format='tf')
    import pickle
    with open(os.path.join(out_dir, f'{model_name}_scalers.pkl'), 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

    return model, history, results



def moe_spike_staged(
    snapshot_df: pd.DataFrame,
    basic_features: list,
    spike_quantile: float = 0.95,
    target_frac: float = 0.20,
    min_factor: float = 1.0,
    max_factor: float = 5.0,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    es_patience: int = 15,
    lr_patience: int = 7,
    model_name: str = 'moe_spike_stages'
):
    """
    Two-stage training:
    1) Train only regression (vol) head with spike head frozen for epochs//2 epochs
    2) Unfreeze spike head and train both heads for remaining epochs
    Returns: model, history_joint, results (same format as baseline)
    """
    seq_df = util.generate_tick_sequences(snapshot_df, feature_cols=basic_features)
    tids = np.sort(seq_df['time_id'].unique())
    split = int(len(tids) * 0.8)
    train_df = seq_df[seq_df['time_id'].isin(tids[:split])].reset_index(drop=True)
    val_df = seq_df[seq_df['time_id'].isin(tids[split:])].reset_index(drop=True)
    spike_thresh = train_df['y'].quantile(spike_quantile)
    train_df['spike'] = (train_df['y'] > spike_thresh).astype(int)
    val_df['spike'] = (val_df['y'] > spike_thresh).astype(int)

    flat_X = np.vstack(train_df['X'].values).astype('float32')
    x_scaler = StandardScaler().fit(flat_X)
    y_scaler = StandardScaler().fit(train_df['y'].values.reshape(-1,1))

    def gen(df, oversample_factor):
        pos_df = df[df['spike']==1]; neg_df = df[df['spike']==0]
        while True:
            neg_shuf = neg_df.sample(frac=1).reset_index(drop=True)
            pos_shuf = pos_df.sample(frac=oversample_factor, replace=True).reset_index(drop=True)
            aug = pd.concat([neg_shuf, pos_shuf]).sample(frac=1).reset_index(drop=True)
            for i in range(0, len(aug), batch_size):
                batch = aug.iloc[i:i+batch_size]
                Xb = np.stack(batch['X'].values).astype('float32')
                Xb_s = x_scaler.transform(Xb.reshape(-1, Xb.shape[-1])).reshape(Xb.shape)
                yb = batch['y'].values.reshape(-1,1).astype('float32')
                yb_s = y_scaler.transform(yb).ravel()
                spk = batch['spike'].values.astype('float32').reshape(-1,1)
                yield Xb_s, {'vol': yb_s, 'spike': spk}
      
    def val_generator(df, batch_size, x_scaler, y_scaler):
        W,D = df.iloc[0]['X'].shape
        while True:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                Xb = np.stack(batch['X'].values).astype('float32')
                Xb = x_scaler.transform(Xb.reshape(-1,D)).reshape(-1,W,D)
                yb = batch['y'].values.reshape(-1,1).astype('float32')
                yb = y_scaler.transform(yb).ravel()
                spk = batch['spike'].values.astype('float32').reshape(-1,1)
                yield Xb, {'vol': yb, 'spike': spk}

    pos = train_df['spike'].sum(); neg = len(train_df)-pos
    raw_factor = (target_frac * neg/(pos*(1-target_frac))) if pos>0 else 1.0
    oversample_factor = int(np.clip(raw_factor, min_factor, max_factor))

    steps = len(train_df)//batch_size
    vsteps = int(np.ceil(len(val_df)/batch_size))
    train_gen = gen(train_df, oversample_factor)
    val_gen = val_generator(val_df, batch_size, x_scaler, y_scaler)
    W,D = train_df.iloc[0]['X'].shape

    inp = Input((W,D))
    h = LSTM(128)(inp)
    spike_prob = Dense(1, activation='sigmoid', name='spike')(h)
    en = Dense(1, name='expert_normal')(h)
    es_ = Dense(1, name='expert_spike')(h)
    vol_out = Lambda(lambda args: (1-args[0])*args[1] + args[0]*args[2], name='vol')([spike_prob,en,es_])
    model = Model(inp, [vol_out, spike_prob])

    def weighted_mse(y_true,y_pred): return K.mean((1+K.abs(y_true)/(K.mean(K.abs(y_true))+K.epsilon()))*K.square(y_true-y_pred))
    def focal_bce(gamma=4,alpha=0.25):
        bce = tf.keras.losses.BinaryCrossentropy()
        def loss(y_true,y_pred): p_t=y_true*y_pred+(1-y_true)*(1-y_pred); return K.mean(alpha*K.pow(1-p_t,gamma)*bce(y_true,y_pred))
        return loss

    model.get_layer('spike').trainable=False
    model.compile(
        optimizer=Adam(learning_rate),
        loss={
        'vol':   weighted_mse,
        'spike': focal_bce()
        },
        loss_weights={
        'vol':   1.0,
        'spike': 0.0
        },
        metrics={
        'vol': [tf.keras.metrics.RootMeanSquaredError(name='rmse')],
        'spike': ['accuracy']
        }
    )

    warmup_steps_stage1 = steps * (epochs//2)
    warmup1 = WarmUp(warmup_steps_stage1, learning_rate)
    ckpt1 = ModelCheckpoint(f'{model_name}_stage1.h5', save_best_only=True,
                            monitor='val_vol_loss', mode='min')
    es1 = EarlyStopping(monitor='val_vol_loss', mode='min',
                        patience=es_patience, restore_best_weights=True)
    rlr1 = ReduceLROnPlateau(monitor='val_vol_loss', mode='min',
                             factor=0.5, patience=lr_patience, min_lr=1e-7)
    hist1 = model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps,
                      validation_steps=vsteps, epochs=epochs//2,
                      callbacks=[warmup1, ckpt1, es1, rlr1],
                      verbose=1)

    model.get_layer('spike').trainable=True
    model.compile(optimizer=Adam(learning_rate),
                  loss={'vol':weighted_mse,'spike':focal_bce()},
                  loss_weights={'vol':1.0,'spike':0.5},
                  metrics={'vol':tf.keras.metrics.RootMeanSquaredError(name='rmse'),'spike':'accuracy'})

    warmup_steps_stage2 = steps * (epochs//2)
    warmup2 = WarmUp(warmup_steps_stage2, learning_rate)
    ckpt2 = ModelCheckpoint(f'{model_name}.h5', save_best_only=True,
                            monitor='val_vol_loss', mode='min')
    es2 = EarlyStopping(monitor='val_vol_loss', mode='min',
                        patience=es_patience, restore_best_weights=True)
    rlr2 = ReduceLROnPlateau(monitor='val_vol_loss', mode='min',
                             factor=0.5, patience=lr_patience, min_lr=1e-7)
    hist2 = model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps,
                      validation_steps=vsteps, epochs=epochs-epochs//2,
                      callbacks=[warmup2, ckpt2, es2, rlr2],
                      verbose=1)

    inf_model=Model(inp,[vol_out,spike_prob,en,es_])
    inf_model.load_weights(os.path.join(f'{model_name}.h5'))
    flat_val=np.vstack(val_df['X'].values).astype('float32')
    Xv=x_scaler.transform(flat_val).reshape(-1,W,D)
    y_true=val_df['y'].values
    vols,probs,norms,spks,times=[],[],[],[],[]
    for seq in Xv:
        t0=time.perf_counter(); v,p,n,s=inf_model.predict(seq[np.newaxis],verbose=0)
        times.append(time.perf_counter()-t0)
        vols.append(v.ravel()[0]); probs.append(p.ravel()[0]); norms.append(n.ravel()[0]); spks.append(s.ravel()[0])
    
    results=pd.DataFrame({'time_id':val_df['time_id'],'start_time':val_df['start_time']})
    results['y_true']=y_true; results['y_pred']=y_scaler.inverse_transform(np.array(vols).reshape(-1,1)).ravel()
    results['spike_prob']=probs; results['expert_normal']=y_scaler.inverse_transform(np.array(norms).reshape(-1,1)).ravel()
    results['expert_spike']=y_scaler.inverse_transform(np.array(spks).reshape(-1,1)).ravel(); results['inference_time']=times

    mse=mean_squared_error(results['y_true'],results['y_pred']); print(f"Calibrated RMSE: {np.sqrt(mse):.8f}")
    ql=util.qlike_loss(results['y_true'],results['y_pred']); print(f"Calibrated QLIKE: {ql:.4f}")
    spike_pred=(results['spike_prob']>spike_thresh).astype(int)
    print("Spike Acc:",accuracy_score(val_df['spike'],spike_pred))
    print("Spike Recall:",recall_score(val_df['spike'],spike_pred))

    out_dir = 'out/lstm'
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, f'{model_name}.h5'), save_format='tf')
    import pickle
    with open(os.path.join(out_dir, f'{model_name}_scalers.pkl'), 'wb') as f:
        pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)

    # Return joint stage history
    return model, hist1, hist2, results
